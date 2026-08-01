/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * OPEN-S4 bounded real-weight comparison.
 *
 * This is not a TinyLlama runtime. It compares selected native BF16
 * safetensors matrices with their corresponding Q16.48 MGW matrices using
 * pinned BF16-valued probe vectors.
 *
 * Build from the repository root:
 *   nvcc -O3 -std=c++17 -arch=sm_120 \
 *     -Xcompiler=-Wall,-Wextra,-Werror,-Wno-unused-function \
 *     -Ispikes/gpu-real-weight -Isrc/llama \
 *     spikes/gpu-real-weight/bench_real_weights.cu \
 *     -o bench_real_weights -lcublas
 *
 * Run only through preflight_and_run.sh after independent source review.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "real_weight_math.h"
#include "safetensors.h"

namespace {

constexpr const char *PUBLIC_SOURCE_COMMIT =
    "1b80e024c5fadea955e3892578a36dbc80a8a0b5";
constexpr std::size_t EXPECTED_ST_BYTES = UINT64_C(2200119864);
constexpr std::size_t EXPECTED_MGW_BYTES = UINT64_C(8800406496);
constexpr uint32_t MGW_ENDIAN_TAG = UINT32_C(0x01020304);
constexpr int INTEGER_BLOCK = 256;
constexpr int TIMING_CYCLES = 4;
constexpr uint32_t BF16_POISON_A = UINT32_C(0x7fc00001);
constexpr uint32_t BF16_POISON_B = UINT32_C(0x7fc00002);
constexpr std::array<std::size_t, 2> PROBE_OFFSETS = {2048, 65536};
constexpr std::array<std::size_t, 2> PROBE_EXPECTED_ROUNDED = {0, 0};

struct TensorSpec {
    const char *id;
    const char *name;
    int rows;
    int cols;
    std::size_t expected_rounded;
    std::size_t expected_rounded_to_zero;
};

constexpr TensorSpec TENSOR_SPECS[] = {
    {"RW0", "model.layers.0.self_attn.k_proj.weight", 256, 2048, 0, 0},
    {"RW1", "model.layers.0.self_attn.q_proj.weight", 2048, 2048, 0, 0},
    {"RW2", "model.layers.0.mlp.gate_proj.weight", 5632, 2048, 0, 0},
    {"RW3", "model.layers.0.mlp.down_proj.weight", 2048, 5632, 0, 0},
    {
        "RW4",
        "model.embed_tokens.weight",
        32000,
        2048,
        225280,
        225280,
    },
};

constexpr const char *EMBEDDING_NAME = "model.embed_tokens.weight";

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void cuda_check(cudaError_t status, const char *expression, int line) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(
            message,
            sizeof(message),
            "CUDA line %d (%s): %s",
            line,
            expression,
            cudaGetErrorString(status)
        );
        fail(message);
    }
}

void cublas_check(cublasStatus_t status, const char *expression, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[512];
        std::snprintf(
            message,
            sizeof(message),
            "cuBLAS line %d (%s): status=%d",
            line,
            expression,
            static_cast<int>(status)
        );
        fail(message);
    }
}

#define CUDA_CHECK(expression) \
    cuda_check((expression), #expression, __LINE__)
#define CUBLAS_CHECK(expression) \
    cublas_check((expression), #expression, __LINE__)

void check_launch(const char *label) {
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        fail(
            std::string("kernel launch ") + label + ": "
            + cudaGetErrorString(status)
        );
    }
}

bool checked_add(std::size_t a, std::size_t b, std::size_t *out) {
    if (out == nullptr || b > std::numeric_limits<std::size_t>::max() - a) {
        return false;
    }
    *out = a + b;
    return true;
}

bool checked_multiply(std::size_t a, std::size_t b, std::size_t *out) {
    if (out == nullptr
        || (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)) {
        return false;
    }
    *out = a * b;
    return true;
}

class MappedFile {
public:
    explicit MappedFile(const char *path) {
        fd_ = open(path, O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) fail(std::string("open failed: ") + std::strerror(errno));
        struct stat status {};
        if (fstat(fd_, &status) != 0 || status.st_size <= 0) {
            const int saved_errno = errno;
            close(fd_);
            fd_ = -1;
            fail(std::string("fstat failed: ") + std::strerror(saved_errno));
        }
        if (static_cast<uintmax_t>(status.st_size)
            > std::numeric_limits<std::size_t>::max()) {
            close(fd_);
            fd_ = -1;
            fail("mapped file is too large for this host");
        }
        size_ = static_cast<std::size_t>(status.st_size);
        mapping_ = static_cast<const uint8_t *>(
            mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0)
        );
        if (mapping_ == MAP_FAILED) {
            const int saved_errno = errno;
            mapping_ = nullptr;
            close(fd_);
            fd_ = -1;
            fail(std::string("mmap failed: ") + std::strerror(saved_errno));
        }
    }

    ~MappedFile() {
        if (mapping_ != nullptr) munmap(const_cast<uint8_t *>(mapping_), size_);
        if (fd_ >= 0) close(fd_);
    }

    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    const uint8_t *data() const { return mapping_; }
    std::size_t size() const { return size_; }

private:
    int fd_ = -1;
    const uint8_t *mapping_ = nullptr;
    std::size_t size_ = 0;
};

struct MgwHeader {
    char magic[4];
    uint32_t version;
    uint32_t endian_tag;
    uint32_t tensor_count;
    uint64_t index_offset;
    uint64_t data_offset;
    uint8_t reserved[32];
};

struct MgwIndexEntry {
    char name[64];
    uint64_t elements;
    uint64_t data_offset;
    uint32_t dimensions;
    uint32_t shape[2];
    uint32_t reserved;
};

static_assert(sizeof(MgwHeader) == 64, "MGW header layout");
static_assert(sizeof(MgwIndexEntry) == 96, "MGW index layout");

class MgwModel {
public:
    explicit MgwModel(const char *path) : file_(path) {
        if (file_.size() != EXPECTED_MGW_BYTES) {
            fail("MGW byte size differs from pinned artifact");
        }
        std::memcpy(&header_, file_.data(), sizeof(header_));
        if (std::memcmp(header_.magic, "MGW\0", 4) != 0
            || header_.version != 1
            || header_.endian_tag != MGW_ENDIAN_TAG) {
            fail("unsupported MGW header");
        }
        if (header_.tensor_count == 0 || header_.tensor_count > 512) {
            fail("MGW tensor count outside bounded range");
        }
        std::size_t index_bytes = 0;
        std::size_t index_end = 0;
        if (!checked_multiply(
                header_.tensor_count, sizeof(MgwIndexEntry), &index_bytes
            )
            || header_.index_offset > file_.size()
            || !checked_add(
                static_cast<std::size_t>(header_.index_offset),
                index_bytes,
                &index_end
            )
            || index_end > file_.size()
            || header_.data_offset < index_end
            || header_.data_offset > file_.size()) {
            fail("MGW index bounds are invalid");
        }

        entries_.resize(header_.tensor_count);
        std::memcpy(
            entries_.data(),
            file_.data() + header_.index_offset,
            index_bytes
        );
        for (const MgwIndexEntry &entry : entries_) validate_entry(entry);
    }

    const MgwIndexEntry &find(const char *name) const {
        const MgwIndexEntry *match = nullptr;
        for (const MgwIndexEntry &entry : entries_) {
            if (std::strncmp(entry.name, name, sizeof(entry.name)) == 0) {
                if (match != nullptr) fail("duplicate MGW tensor name");
                match = &entry;
            }
        }
        if (match == nullptr) fail(std::string("missing MGW tensor: ") + name);
        return *match;
    }

    const int64_t *data(const MgwIndexEntry &entry) const {
        return reinterpret_cast<const int64_t *>(
            file_.data() + entry.data_offset
        );
    }

private:
    void validate_entry(const MgwIndexEntry &entry) const {
        if (std::memchr(entry.name, '\0', sizeof(entry.name)) == nullptr) {
            fail("MGW tensor name is not terminated");
        }
        if (entry.dimensions < 1 || entry.dimensions > 2
            || entry.elements == 0
            || entry.shape[0] == 0
            || (entry.dimensions == 2 && entry.shape[1] == 0)
            || (entry.dimensions == 1 && entry.shape[1] != 0)
            || entry.data_offset % alignof(int64_t) != 0) {
            fail("MGW tensor metadata is invalid");
        }
        const uint64_t shape_elements =
            static_cast<uint64_t>(entry.shape[0])
            * (entry.dimensions == 2 ? entry.shape[1] : 1u);
        if (shape_elements != entry.elements) {
            fail("MGW tensor shape/count mismatch");
        }
        std::size_t bytes = 0;
        std::size_t end = 0;
        if (entry.elements > std::numeric_limits<std::size_t>::max()
            || !checked_multiply(
                static_cast<std::size_t>(entry.elements),
                sizeof(int64_t),
                &bytes
            )
            || entry.data_offset < header_.data_offset
            || entry.data_offset > file_.size()
            || !checked_add(
                static_cast<std::size_t>(entry.data_offset), bytes, &end
            )
            || end > file_.size()) {
            fail("MGW tensor data bounds are invalid");
        }
    }

    MappedFile file_;
    MgwHeader header_{};
    std::vector<MgwIndexEntry> entries_;
};

class SafetensorModel {
public:
    explicit SafetensorModel(const char *path) {
        struct stat status {};
        if (stat(path, &status) != 0
            || static_cast<std::size_t>(status.st_size) != EXPECTED_ST_BYTES) {
            fail("safetensors byte size differs from pinned artifact");
        }
        st_model_init(&model_);
        if (st_model_load_shard(&model_, path) != 0) {
            st_model_free(&model_);
            fail("failed to parse safetensors");
        }
    }

    ~SafetensorModel() { st_model_free(&model_); }

    SafetensorModel(const SafetensorModel &) = delete;
    SafetensorModel &operator=(const SafetensorModel &) = delete;

    const st_tensor_t &find(const char *name) {
        st_tensor_t *match = nullptr;
        for (int index = 0; index < model_.num_tensors; ++index) {
            if (std::strcmp(model_.tensors[index].name, name) == 0) {
                if (match != nullptr) fail("duplicate safetensors name");
                match = &model_.tensors[index];
            }
        }
        if (match == nullptr) {
            fail(std::string("missing safetensors tensor: ") + name);
        }
        if (st_tensor_validate_bounds(&model_, match) != 0) {
            fail("safetensors tensor bounds are invalid");
        }
        return *match;
    }

    const uint16_t *data(const st_tensor_t &tensor) {
        return reinterpret_cast<const uint16_t *>(
            st_tensor_data(&model_, const_cast<st_tensor_t *>(&tensor))
        );
    }

private:
    st_model_t model_{};
};

struct TensorView {
    TensorSpec spec;
    const uint16_t *bf16;
    const int64_t *q48;
    std::size_t elements;
};

TensorView bind_tensor(
    SafetensorModel &safetensors,
    const MgwModel &mgw,
    const TensorSpec &spec
) {
    const st_tensor_t &floating = safetensors.find(spec.name);
    const MgwIndexEntry &integer = mgw.find(spec.name);
    const uint64_t expected_elements =
        static_cast<uint64_t>(spec.rows) * spec.cols;
    if (floating.dtype != ST_DTYPE_BF16
        || floating.ndim != 2
        || floating.shape[0] != spec.rows
        || floating.shape[1] != spec.cols
        || floating.num_elements != static_cast<int64_t>(expected_elements)
        || floating.data_size != expected_elements * sizeof(uint16_t)) {
        fail(std::string("safetensors metadata mismatch: ") + spec.name);
    }
    if (integer.dimensions != 2
        || integer.shape[0] != static_cast<uint32_t>(spec.rows)
        || integer.shape[1] != static_cast<uint32_t>(spec.cols)
        || integer.elements != expected_elements) {
        fail(std::string("MGW metadata mismatch: ") + spec.name);
    }
    return {
        spec,
        safetensors.data(floating),
        mgw.data(integer),
        static_cast<std::size_t>(expected_elements),
    };
}

void verify_tensor_identity(const TensorView &tensor) {
    std::size_t mismatch_count = 0;
    std::size_t rounded_count = 0;
    std::size_t rounded_to_zero = 0;
    std::size_t first_mismatch = 0;
    int64_t first_converted = 0;
    int64_t first_mgw = 0;
    for (std::size_t index = 0; index < tensor.elements; ++index) {
        int64_t converted = 0;
        if (!real_weight::bf16_to_q48(tensor.bf16[index], &converted)) {
            fail(
                std::string("BF16 is outside the repository conversion domain: ")
                + tensor.spec.name
            );
        }
        const int64_t repository_converted =
            bfloat16_to_q1648(tensor.bf16[index]);
        if (converted != repository_converted) {
            fail(
                std::string("safe BF16 conversion differs from pinned loader: ")
                + tensor.spec.name
            );
        }
        if (!real_weight::bf16_is_exact_q48(tensor.bf16[index])) {
            ++rounded_count;
            if (converted == 0) ++rounded_to_zero;
        }
        if (repository_converted != tensor.q48[index]) {
            if (mismatch_count == 0) {
                first_mismatch = index;
                first_converted = repository_converted;
                first_mgw = tensor.q48[index];
            }
            ++mismatch_count;
        }
    }
    std::printf(
        "IDENTITY tensor_id=%s elements=%zu mismatches=%zu "
        "rounded_elements=%zu rounded_to_zero=%zu status=%s\n",
        tensor.spec.id,
        tensor.elements,
        mismatch_count,
        rounded_count,
        rounded_to_zero,
        mismatch_count == 0
                && rounded_count == tensor.spec.expected_rounded
                && rounded_to_zero == tensor.spec.expected_rounded_to_zero
            ? "PASS"
            : "FAIL"
    );
    if (mismatch_count != 0) {
        char message[512];
        std::snprintf(
            message,
            sizeof(message),
            "%s first weight mismatch index=%zu converted=%lld mgw=%lld",
            tensor.spec.id,
            first_mismatch,
            static_cast<long long>(first_converted),
            static_cast<long long>(first_mgw)
        );
        fail(message);
    }
    if (rounded_count != tensor.spec.expected_rounded
        || rounded_to_zero != tensor.spec.expected_rounded_to_zero) {
        fail(
            std::string("repository-conversion rounding census changed: ")
            + tensor.spec.id
        );
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ == 0
            || count_
                > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            fail("invalid device allocation count");
        }
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&pointer_), count_ * sizeof(T)
        ));
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) cudaFree(pointer_);
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get() { return pointer_; }
    const T *get() const { return pointer_; }
    std::size_t count() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T *pointer_ = nullptr;
    std::size_t count_ = 0;
};

class EventPair {
public:
    EventPair() {
        CUDA_CHECK(cudaEventCreate(&start_));
        const cudaError_t status = cudaEventCreate(&stop_);
        if (status != cudaSuccess) {
            cudaEventDestroy(start_);
            start_ = nullptr;
            CUDA_CHECK(status);
        }
    }
    ~EventPair() {
        if (start_ != nullptr) cudaEventDestroy(start_);
        if (stop_ != nullptr) cudaEventDestroy(stop_);
    }
    cudaEvent_t start() const { return start_; }
    cudaEvent_t stop() const { return stop_; }
    void close_checked() {
        const cudaError_t stop_status =
            stop_ == nullptr ? cudaSuccess : cudaEventDestroy(stop_);
        const cudaError_t start_status =
            start_ == nullptr ? cudaSuccess : cudaEventDestroy(start_);
        stop_ = nullptr;
        start_ = nullptr;
        CUDA_CHECK(stop_status);
        CUDA_CHECK(start_status);
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

class CublasHandle {
public:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
        const cublasStatus_t status =
            cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
        if (status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle_);
            handle_ = nullptr;
            CUBLAS_CHECK(status);
        }
    }
    ~CublasHandle() {
        if (handle_ != nullptr) cublasDestroy(handle_);
    }
    cublasHandle_t get() const { return handle_; }
    void close_checked() {
        const cublasStatus_t status = handle_ == nullptr
            ? CUBLAS_STATUS_SUCCESS
            : cublasDestroy(handle_);
        handle_ = nullptr;
        CUBLAS_CHECK(status);
    }

private:
    cublasHandle_t handle_ = nullptr;
};

struct I128Pair {
    uint64_t low;
    int64_t high;
};

__device__ __forceinline__ void add_product(
    uint64_t &accumulator_low,
    int64_t &accumulator_high,
    int64_t left,
    int64_t right
) {
    const uint64_t product_low =
        static_cast<uint64_t>(left) * static_cast<uint64_t>(right);
    int64_t product_high;
    asm volatile(
        "mul.hi.s64 %0, %1, %2;"
        : "=l"(product_high)
        : "l"(left), "l"(right)
    );
    asm volatile(
        "add.cc.u64 %0, %0, %2;\n\t"
        "addc.s64   %1, %1, %3;"
        : "+l"(accumulator_low), "+l"(accumulator_high)
        : "l"(product_low), "l"(product_high)
    );
}

__device__ __forceinline__ void add_pair(
    uint64_t &accumulator_low,
    int64_t &accumulator_high,
    const I128Pair other
) {
    asm volatile(
        "add.cc.u64 %0, %0, %2;\n\t"
        "addc.s64   %1, %1, %3;"
        : "+l"(accumulator_low), "+l"(accumulator_high)
        : "l"(other.low), "l"(other.high)
    );
}

__global__ void q48_gemv_kernel(
    const int64_t *__restrict__ weights,
    const int64_t *__restrict__ input,
    uint64_t *__restrict__ output_bits,
    int rows,
    int cols
) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const int64_t *weight_row =
        weights + static_cast<std::size_t>(row) * cols;

    uint64_t accumulator_low = 0;
    int64_t accumulator_high = 0;
    for (int column = static_cast<int>(threadIdx.x);
         column < cols;
         column += static_cast<int>(blockDim.x)) {
        add_product(
            accumulator_low,
            accumulator_high,
            input[column],
            weight_row[column]
        );
    }

    extern __shared__ I128Pair partials[];
    partials[threadIdx.x] = {accumulator_low, accumulator_high};
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            add_pair(
                partials[threadIdx.x].low,
                partials[threadIdx.x].high,
                partials[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output_bits[row] =
            (static_cast<uint64_t>(partials[0].high) << 16)
            | (partials[0].low >> real_weight::FRACTION_BITS);
    }
}

__global__ void l2_pressure_kernel(uint64_t *data, std::size_t count) {
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (; index < count; index += stride) {
        const uint64_t value = data[index];
        data[index] = value + index + 1;
    }
}

__global__ void fill_u32_kernel(
    uint32_t *data,
    std::size_t count,
    uint32_t value
) {
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (; index < count; index += stride) data[index] = value;
}

void fill_bf16_output(float *output, std::size_t count, uint32_t poison_bits) {
    if (output == nullptr || count == 0) fail("invalid BF16 poison target");
    const std::size_t blocks = (count + 255) / 256;
    fill_u32_kernel<<<static_cast<unsigned>(blocks), 256>>>(
        reinterpret_cast<uint32_t *>(output), count, poison_bits
    );
    check_launch("fill_bf16_output");
}

void launch_integer(
    const int64_t *weights,
    const int64_t *input,
    uint64_t *output,
    int rows,
    int cols
) {
    q48_gemv_kernel<<<rows, INTEGER_BLOCK, INTEGER_BLOCK * sizeof(I128Pair)>>>(
        weights, input, output, rows, cols
    );
    check_launch("q48_gemv");
}

void launch_bf16(
    cublasHandle_t handle,
    const uint16_t *weights,
    const uint16_t *input,
    float *output,
    int rows,
    int cols
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        rows,
        1,
        cols,
        &alpha,
        weights,
        CUDA_R_16BF,
        cols,
        input,
        CUDA_R_16BF,
        cols,
        &beta,
        output,
        CUDA_R_32F,
        rows,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

int64_t output_bits_to_i64(uint64_t bits) {
    int64_t value;
    static_assert(sizeof(value) == sizeof(bits), "int64 representation");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float output_bits_to_float(uint32_t bits) {
    float value;
    static_assert(sizeof(value) == sizeof(bits), "float32 representation");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

class Bf16OracleTable {
public:
    Bf16OracleTable() {
        for (std::size_t raw = 0; raw < components_.size(); ++raw) {
            int64_t production = 0;
            int64_t independent = 0;
            const bool production_ok = real_weight::bf16_to_q48(
                static_cast<uint16_t>(raw), &production
            );
            const bool independent_ok =
                real_weight::bf16_to_q48_independent(
                    static_cast<uint16_t>(raw), &independent
                );
            if (production_ok != independent_ok
                || (production_ok && production != independent)) {
                fail("BF16 repository conversion self-test mismatch");
            }
            real_weight::Bf16Component component{};
            const bool component_ok = real_weight::bf16_component(
                static_cast<uint16_t>(raw), &component
            );
            const bool finite =
                (raw & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
            if (component_ok != finite) {
                fail("BF16 rational component domain mismatch");
            }
            valid_[raw] = component_ok ? 1u : 0u;
            components_[raw] = component;
        }
        std::puts(
            "SELF_TEST name=bf16_safe_conversion_and_component_domain_exhaustive "
            "status=PASS"
        );
    }

    const real_weight::Bf16Component &component(uint16_t bits) const {
        if (valid_[bits] == 0u) {
            fail("native BF16 value is outside the rational oracle");
        }
        return components_[bits];
    }

private:
    std::array<real_weight::Bf16Component, 65536> components_{};
    std::array<uint8_t, 65536> valid_{};
};

std::vector<real_weight::DotReference> make_integer_references(
    const TensorView &tensor,
    const int64_t *input
) {
    std::vector<real_weight::DotReference> references(tensor.spec.rows);
    for (int row = 0; row < tensor.spec.rows; ++row) {
        if (!real_weight::dot_reference(
                input,
                tensor.q48
                    + static_cast<std::size_t>(row) * tensor.spec.cols,
                static_cast<std::size_t>(tensor.spec.cols),
                &references[row]
            )) {
            fail(
                std::string("CPU reference overflow/range failure: ")
                + tensor.spec.id
            );
        }
    }
    return references;
}

std::vector<real_weight::FloatingReference> make_bf16_references(
    const TensorView &tensor,
    const uint16_t *input,
    const Bf16OracleTable &oracle
) {
    std::vector<real_weight::Bf16Component> decoded_input(
        static_cast<std::size_t>(tensor.spec.cols)
    );
    for (int column = 0; column < tensor.spec.cols; ++column) {
        decoded_input[static_cast<std::size_t>(column)] =
            oracle.component(input[column]);
    }

    std::vector<real_weight::FloatingReference> references(
        static_cast<std::size_t>(tensor.spec.rows)
    );
    std::vector<real_weight::Bf16Component> decoded_weights(
        static_cast<std::size_t>(tensor.spec.cols)
    );
    for (int row = 0; row < tensor.spec.rows; ++row) {
        const uint16_t *weight_row =
            tensor.bf16
            + static_cast<std::size_t>(row) * tensor.spec.cols;
        for (int column = 0; column < tensor.spec.cols; ++column) {
            decoded_weights[static_cast<std::size_t>(column)] =
                oracle.component(weight_row[column]);
        }
        if (!real_weight::bf16_component_dot_reference(
                decoded_input.data(),
                decoded_weights.data(),
                decoded_input.size(),
                &references[static_cast<std::size_t>(row)]
            )) {
            fail(
                std::string("independent BF16 rational oracle failed: ")
                + tensor.spec.id
            );
        }
    }
    return references;
}

std::vector<int64_t> run_integer_once(
    const DeviceBuffer<int64_t> &weights,
    const DeviceBuffer<int64_t> &input,
    DeviceBuffer<uint64_t> &output,
    int rows,
    int cols,
    unsigned char poison
) {
    CUDA_CHECK(cudaMemset(output.get(), poison, output.bytes()));
    launch_integer(weights.get(), input.get(), output.get(), rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint64_t> bits(static_cast<std::size_t>(rows));
    CUDA_CHECK(cudaMemcpy(
        bits.data(),
        output.get(),
        output.bytes(),
        cudaMemcpyDeviceToHost
    ));
    std::vector<int64_t> result(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        result[index] = output_bits_to_i64(bits[index]);
    }
    return result;
}

std::vector<float> run_bf16_once(
    cublasHandle_t handle,
    const DeviceBuffer<uint16_t> &weights,
    const DeviceBuffer<uint16_t> &input,
    DeviceBuffer<float> &output,
    int rows,
    int cols,
    uint32_t poison_bits
) {
    fill_bf16_output(output.get(), output.count(), poison_bits);
    launch_bf16(handle, weights.get(), input.get(), output.get(), rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> result(static_cast<std::size_t>(rows));
    CUDA_CHECK(cudaMemcpy(
        result.data(),
        output.get(),
        output.bytes(),
        cudaMemcpyDeviceToHost
    ));
    return result;
}

void require_integer_outputs(
    const char *tensor_id,
    int probe,
    const std::vector<int64_t> &actual,
    const std::vector<real_weight::DotReference> &expected
) {
    if (actual.size() != expected.size()) fail("integer output size mismatch");
    const real_weight::IntegerVerdict verdict =
        real_weight::verify_integer_values(
            actual.data(), expected.data(), actual.size()
        );
    std::printf(
        "INT_VERIFY tensor_id=%s probe=%d elements=%zu mismatches=%zu "
        "status=%s\n",
        tensor_id,
        probe,
        actual.size(),
        verdict.mismatches,
        verdict.passed() ? "PASS" : "FAIL"
    );
    if (!verdict.passed()) {
        char message[512];
        std::snprintf(
            message,
            sizeof(message),
            "%s probe=%d first integer mismatch index=%zu actual=%lld "
            "expected=%lld",
            tensor_id,
            probe,
            verdict.first_mismatch,
            static_cast<long long>(actual[verdict.first_mismatch]),
            static_cast<long long>(
                expected[verdict.first_mismatch].q48
            )
        );
        fail(message);
    }
}

void require_integer_poison_distinct(
    const std::vector<real_weight::DotReference> &expected,
    unsigned char poison,
    const char *context
) {
    const uint64_t repeated =
        static_cast<uint64_t>(poison) * UINT64_C(0x0101010101010101);
    const int64_t poison_value = output_bits_to_i64(repeated);
    for (const real_weight::DotReference &reference : expected) {
        if (reference.q48 == poison_value) {
            fail(
                std::string("integer poison collides with expected output: ")
                + context
            );
        }
    }
}

void require_bf16_outputs(
    const char *tensor_id,
    int probe,
    int cols,
    const std::vector<float> &actual,
    const std::vector<real_weight::FloatingReference> &expected,
    const std::vector<real_weight::DotReference> &integer_expected,
    uint32_t forbidden_bits
) {
    if (actual.size() != expected.size()
        || actual.size() != integer_expected.size()) {
        fail("BF16 output/reference size mismatch");
    }
    const real_weight::FloatingVerdict verdict =
        real_weight::verify_floating_values(
            actual.data(),
            expected.data(),
            actual.size(),
            static_cast<std::size_t>(cols),
            forbidden_bits,
            true
        );
    long double maximum_cross_lane = 0.0L;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const long double observed = static_cast<long double>(actual[index]);
        const long double integer_real =
            static_cast<long double>(integer_expected[index].q48)
            / static_cast<long double>(real_weight::Q48_SCALE);
        maximum_cross_lane = std::max(
            maximum_cross_lane, std::fabs(observed - integer_real)
        );
    }
    std::printf(
        "BF16_VERIFY tensor_id=%s probe=%d elements=%zu violations=%zu "
        "sentinel_hits=%zu "
        "max_abs_error=%.12Le max_bound=%.12Le "
        "max_cross_lane=%.12Le status=%s\n",
        tensor_id,
        probe,
        actual.size(),
        verdict.violations,
        verdict.sentinel_hits,
        verdict.maximum_error,
        verdict.maximum_bound,
        maximum_cross_lane,
        verdict.passed() ? "PASS" : "FAIL"
    );
    if (!verdict.passed()) {
        fail(
            std::string(tensor_id) + " probe=" + std::to_string(probe)
            + " BF16 error-bound failure at index="
            + std::to_string(verdict.first_violation)
        );
    }
}

void run_bf16_layout_self_test(cublasHandle_t handle) {
    constexpr int rows = 3;
    constexpr int cols = 5;
    const uint16_t weights[] = {
        0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0,
        0xbf80, 0x3f00, 0xbf00, 0x4000, 0xc000,
        0x3e80, 0xbe80, 0x3f80, 0xbf80, 0x3f00,
    };
    const uint16_t input[] = {0x3f80, 0xbf80, 0x4000, 0x3f00, 0xc000};
    const std::vector<real_weight::FloatingReference> expected = {
        {-3.0L, 24.0L},
        {2.5L, 8.5L},
        {1.0L, 4.5L},
    };
    const std::vector<real_weight::DotReference> integer_expected = {
        {-3 * static_cast<int64_t>(real_weight::Q48_SCALE), -3.0L, 24.0L},
        {
            5 * static_cast<int64_t>(real_weight::Q48_SCALE / 2),
            2.5L,
            8.5L,
        },
        {static_cast<int64_t>(real_weight::Q48_SCALE), 1.0L, 4.5L},
    };
    DeviceBuffer<uint16_t> device_weights(rows * cols);
    DeviceBuffer<uint16_t> device_input(cols);
    DeviceBuffer<float> device_output(rows);
    CUDA_CHECK(cudaMemcpy(
        device_weights.get(),
        weights,
        sizeof(weights),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        device_input.get(), input, sizeof(input), cudaMemcpyHostToDevice
    ));
    const std::vector<float> first = run_bf16_once(
        handle,
        device_weights,
        device_input,
        device_output,
        rows,
        cols,
        BF16_POISON_A
    );
    const std::vector<float> second = run_bf16_once(
        handle,
        device_weights,
        device_input,
        device_output,
        rows,
        cols,
        BF16_POISON_B
    );
    require_bf16_outputs(
        "SELF_BF16",
        0,
        cols,
        first,
        expected,
        integer_expected,
        BF16_POISON_A
    );
    require_bf16_outputs(
        "SELF_BF16",
        1,
        cols,
        second,
        expected,
        integer_expected,
        BF16_POISON_B
    );
    if (std::memcmp(
            first.data(), second.data(), first.size() * sizeof(float)
        ) != 0) {
        fail("BF16 layout self-test is not bit-stable");
    }

    std::vector<float> corrupt = first;
    corrupt[0] += 1.0f;
    if (real_weight::verify_floating_values(
            corrupt.data(),
            expected.data(),
            corrupt.size(),
            cols,
            BF16_POISON_A,
            true
        ).passed()) {
        fail("BF16 production verifier accepted corrupted output");
    }
    std::vector<float> skipped(
        first.size(), output_bits_to_float(BF16_POISON_A)
    );
    if (real_weight::verify_floating_values(
            skipped.data(),
            expected.data(),
            skipped.size(),
            cols,
            BF16_POISON_A,
            true
        ).passed()) {
        fail("BF16 production verifier accepted skipped poisoned output");
    }
    std::vector<float> partial = first;
    partial.back() = output_bits_to_float(BF16_POISON_B);
    if (real_weight::verify_floating_values(
            partial.data(),
            expected.data(),
            partial.size(),
            cols,
            BF16_POISON_B,
            true
        ).passed()) {
        fail("BF16 production verifier accepted partial-row output");
    }
    std::puts(
        "SELF_TEST name=bf16_layout_and_production_verifier_controls "
        "status=PASS"
    );
}

void run_integer_self_test() {
    constexpr int rows = 7;
    constexpr int cols = 257;
    constexpr int64_t a = (INT64_C(1) << 32) - 1;
    std::vector<int64_t> input(cols, 0);
    std::vector<int64_t> weights(
        static_cast<std::size_t>(rows) * cols, 0
    );
    input[0] = a;
    input[1] = a;
    input[256] = a;
    input[2] = 1;
    input[3] = 1;
    input[4] = std::numeric_limits<int64_t>::min();
    input[5] = std::numeric_limits<int64_t>::max();
    input[6] = std::numeric_limits<int64_t>::max();
    input[7] = std::numeric_limits<int64_t>::max();

    weights[0 * cols + 0] = a;
    weights[0 * cols + 1] = a;
    weights[1 * cols + 0] = a;
    weights[1 * cols + 256] = a;
    weights[2 * cols + 2] = -1;
    weights[3 * cols + 3] =
        -static_cast<int64_t>(real_weight::Q48_SCALE + 1);
    weights[4 * cols + 4] =
        static_cast<int64_t>(real_weight::Q48_SCALE);
    weights[5 * cols + 5] =
        static_cast<int64_t>(real_weight::Q48_SCALE);
    weights[6 * cols + 6] =
        static_cast<int64_t>(real_weight::Q48_SCALE);
    weights[6 * cols + 7] =
        -static_cast<int64_t>(real_weight::Q48_SCALE);

    const int64_t pinned_expected[] = {
        131071,
        131071,
        -1,
        -2,
        std::numeric_limits<int64_t>::min(),
        std::numeric_limits<int64_t>::max(),
        0,
    };
    TensorView view{
        {"SELF", "self", rows, cols, 0, 0},
        nullptr,
        weights.data(),
        weights.size(),
    };
    const auto reference = make_integer_references(view, input.data());
    for (int row = 0; row < rows; ++row) {
        if (reference[static_cast<std::size_t>(row)].q48
            != pinned_expected[row]) {
            fail("CPU signed-128 oracle differs from pinned integer answer");
        }
    }
    require_integer_poison_distinct(reference, 0xa5, "self-test poison A");
    require_integer_poison_distinct(reference, 0x5a, "self-test poison B");
    DeviceBuffer<int64_t> device_weights(weights.size());
    DeviceBuffer<int64_t> device_input(input.size());
    DeviceBuffer<uint64_t> device_output(rows);
    CUDA_CHECK(cudaMemcpy(
        device_weights.get(),
        weights.data(),
        weights.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        device_input.get(),
        input.data(),
        input.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice
    ));
    const auto first = run_integer_once(
        device_weights,
        device_input,
        device_output,
        rows,
        cols,
        0xa5
    );
    const auto second = run_integer_once(
        device_weights,
        device_input,
        device_output,
        rows,
        cols,
        0x5a
    );
    require_integer_outputs("SELF_INT", 0, first, reference);
    if (first != second) fail("integer self-test is nondeterministic");

    std::vector<int64_t> corrupt = first;
    corrupt.back() ^= 1;
    if (real_weight::verify_integer_values(
            corrupt.data(), reference.data(), corrupt.size()
        ).passed()) {
        fail("integer production verifier accepted corruption");
    }
    const int64_t poison = output_bits_to_i64(
        UINT64_C(0xa5a5a5a5a5a5a5a5)
    );
    std::vector<int64_t> skipped(first.size(), poison);
    if (real_weight::verify_integer_values(
            skipped.data(), reference.data(), skipped.size()
        ).passed()) {
        fail("integer production verifier accepted skipped poisoned output");
    }
    std::vector<int64_t> partial = first;
    partial.back() = poison;
    if (real_weight::verify_integer_values(
            partial.data(), reference.data(), partial.size()
        ).passed()) {
        fail("integer production verifier accepted partial-row output");
    }
    std::puts(
        "SELF_TEST name=integer_257col_carry_floor_boundaries_and_verifier "
        "status=PASS"
    );
}

void apply_l2_pressure(DeviceBuffer<uint64_t> &buffer) {
    l2_pressure_kernel<<<4096, 256>>>(buffer.get(), buffer.count());
    check_launch("l2_pressure");
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct TimedInteger {
    float microseconds;
    std::vector<int64_t> output;
};

struct TimedBf16 {
    float microseconds;
    std::vector<float> output;
};

TimedInteger measure_integer(
    const DeviceBuffer<int64_t> &weights,
    const DeviceBuffer<int64_t> &input,
    DeviceBuffer<uint64_t> &output,
    int rows,
    int cols,
    unsigned char poison
) {
    CUDA_CHECK(cudaMemset(output.get(), poison, output.bytes()));
    EventPair events;
    CUDA_CHECK(cudaEventRecord(events.start()));
    launch_integer(weights.get(), input.get(), output.get(), rows, cols);
    CUDA_CHECK(cudaEventRecord(events.stop()));
    CUDA_CHECK(cudaEventSynchronize(events.stop()));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds, events.start(), events.stop()
    ));
    std::vector<uint64_t> bits(static_cast<std::size_t>(rows));
    CUDA_CHECK(cudaMemcpy(
        bits.data(),
        output.get(),
        output.bytes(),
        cudaMemcpyDeviceToHost
    ));
    events.close_checked();
    std::vector<int64_t> values(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        values[index] = output_bits_to_i64(bits[index]);
    }
    return {1000.0f * milliseconds, std::move(values)};
}

TimedBf16 measure_bf16(
    cublasHandle_t handle,
    const DeviceBuffer<uint16_t> &weights,
    const DeviceBuffer<uint16_t> &input,
    DeviceBuffer<float> &output,
    int rows,
    int cols,
    uint32_t poison_bits
) {
    fill_bf16_output(output.get(), output.count(), poison_bits);
    EventPair events;
    CUDA_CHECK(cudaEventRecord(events.start()));
    launch_bf16(
        handle, weights.get(), input.get(), output.get(), rows, cols
    );
    CUDA_CHECK(cudaEventRecord(events.stop()));
    CUDA_CHECK(cudaEventSynchronize(events.stop()));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds, events.start(), events.stop()
    ));
    std::vector<float> values(static_cast<std::size_t>(rows));
    CUDA_CHECK(cudaMemcpy(
        values.data(),
        output.get(),
        output.bytes(),
        cudaMemcpyDeviceToHost
    ));
    events.close_checked();
    return {1000.0f * milliseconds, std::move(values)};
}

double median(std::vector<float> values) {
    if (values.empty()) fail("empty timing sample set");
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) return values[middle];
    return 0.5 * (values[middle - 1] + values[middle]);
}

void run_timing_regime(
    const TensorView &tensor,
    cublasHandle_t handle,
    const DeviceBuffer<uint16_t> &bf16_weights,
    const DeviceBuffer<uint16_t> &bf16_input,
    DeviceBuffer<float> &bf16_output,
    const DeviceBuffer<int64_t> &integer_weights,
    const DeviceBuffer<int64_t> &integer_input,
    DeviceBuffer<uint64_t> &integer_output,
    DeviceBuffer<uint64_t> &pressure,
    const std::vector<real_weight::DotReference> &integer_reference,
    const std::vector<real_weight::FloatingReference> &bf16_reference,
    std::size_t probe_offset,
    const char *regime,
    bool use_pressure
) {
    const char *schedule = "ABBABAAB";
    std::vector<float> bf16_samples;
    std::vector<float> integer_samples;
    require_integer_poison_distinct(
        integer_reference, 0xa5, "timed poison A"
    );
    require_integer_poison_distinct(
        integer_reference, 0x5a, "timed poison B"
    );

    if (!use_pressure) {
        for (int warmup = 0; warmup < 5; ++warmup) {
            launch_bf16(
                handle,
                bf16_weights.get(),
                bf16_input.get(),
                bf16_output.get(),
                tensor.spec.rows,
                tensor.spec.cols
            );
            launch_integer(
                integer_weights.get(),
                integer_input.get(),
                integer_output.get(),
                tensor.spec.rows,
                tensor.spec.cols
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int order_index = 0;
    for (int cycle = 0; cycle < TIMING_CYCLES; ++cycle) {
        for (const char *position = schedule; *position != '\0'; ++position) {
            if (use_pressure) apply_l2_pressure(pressure);
            float microseconds;
            const char *lane;
            const unsigned char integer_poison =
                (order_index & 1) == 0 ? 0xa5 : 0x5a;
            const uint32_t bf16_poison =
                (order_index & 1) == 0 ? BF16_POISON_A : BF16_POISON_B;
            try {
                if (*position == 'A') {
                    lane = "BF16_CUBLAS";
                    TimedBf16 timed = measure_bf16(
                        handle,
                        bf16_weights,
                        bf16_input,
                        bf16_output,
                        tensor.spec.rows,
                        tensor.spec.cols,
                        bf16_poison
                    );
                    if (timed.output.size() != bf16_reference.size()) {
                        fail("timed BF16 output size mismatch");
                    }
                    const real_weight::FloatingVerdict verdict =
                        real_weight::verify_floating_values(
                            timed.output.data(),
                            bf16_reference.data(),
                            timed.output.size(),
                            static_cast<std::size_t>(tensor.spec.cols),
                            bf16_poison,
                            true
                        );
                    if (!verdict.passed()) {
                        fail("timed BF16 output verification failed");
                    }
                    microseconds = timed.microseconds;
                    bf16_samples.push_back(microseconds);
                } else {
                    lane = "Q48_INTEGER";
                    TimedInteger timed = measure_integer(
                        integer_weights,
                        integer_input,
                        integer_output,
                        tensor.spec.rows,
                        tensor.spec.cols,
                        integer_poison
                    );
                    if (timed.output.size() != integer_reference.size()) {
                        fail("timed integer output size mismatch");
                    }
                    const real_weight::IntegerVerdict verdict =
                        real_weight::verify_integer_values(
                            timed.output.data(),
                            integer_reference.data(),
                            timed.output.size()
                        );
                    if (!verdict.passed()) {
                        fail("timed integer output verification failed");
                    }
                    microseconds = timed.microseconds;
                    integer_samples.push_back(microseconds);
                }
            } catch (const std::exception &error) {
                fail(
                    std::string("timing tensor=") + tensor.spec.id
                    + " regime=" + regime
                    + " order_index=" + std::to_string(order_index)
                    + ": " + error.what()
                );
            }
            std::printf(
                "SAMPLE tensor_id=%s regime=%s order_index=%d lane=%s "
                "probe_offset=%zu inner=1 output_verified=true "
                "microseconds=%.6f\n",
                tensor.spec.id,
                regime,
                order_index++,
                lane,
                probe_offset,
                microseconds
            );
        }
    }

    const double bf16_median = median(bf16_samples);
    const double integer_median = median(integer_samples);
    const double bf16_bytes =
        static_cast<double>(tensor.elements) * sizeof(uint16_t);
    const double integer_bytes =
        static_cast<double>(tensor.elements) * sizeof(int64_t);
    std::printf(
        "TIMING_SUMMARY tensor_id=%s regime=%s samples_per_lane=%zu "
        "probe_offset=%zu bf16_weight_bytes=%.0f integer_weight_bytes=%.0f "
        "bf16_median_us=%.6f integer_median_us=%.6f "
        "integer_over_bf16=%.6f bf16_weight_GBs=%.6f "
        "integer_weight_GBs=%.6f regime_comparison=NOT_VALID\n",
        tensor.spec.id,
        regime,
        bf16_samples.size(),
        probe_offset,
        bf16_bytes,
        integer_bytes,
        bf16_median,
        integer_median,
        integer_median / bf16_median,
        bf16_bytes / bf16_median / 1000.0,
        integer_bytes / integer_median / 1000.0
    );
}

void preflight_device_memory(
    const TensorView &tensor,
    std::size_t pressure_bytes
) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::size_t tensor_bytes = 0;
    std::size_t required = pressure_bytes;
    if (!checked_multiply(tensor.elements, 10, &tensor_bytes)
        || !checked_add(required, tensor_bytes, &required)
        || !checked_add(
            required,
            static_cast<std::size_t>(tensor.spec.cols) * 10,
            &required
        )
        || !checked_add(
            required,
            static_cast<std::size_t>(tensor.spec.rows) * 12,
            &required
        )) {
        fail("device-memory requirement overflow");
    }
    constexpr std::size_t margin = 512ULL * 1024 * 1024;
    if (required > 8ULL * 1024 * 1024 * 1024
        || free_bytes < margin
        || required > free_bytes - margin) {
        fail("device-memory preflight failed");
    }
    std::printf(
        "MEMORY_PREFLIGHT tensor_id=%s explicit_allocation_bytes=%zu "
        "free_bytes=%zu total_bytes=%zu status=PASS\n",
        tensor.spec.id,
        required,
        free_bytes,
        total_bytes
    );
}

void run_tensor(
    const TensorView &tensor,
    const TensorView &embedding,
    cublasHandle_t handle,
    DeviceBuffer<uint64_t> &pressure,
    const Bf16OracleTable &bf16_oracle
) {
    preflight_device_memory(tensor, pressure.bytes());
    DeviceBuffer<uint16_t> bf16_weights(tensor.elements);
    DeviceBuffer<int64_t> integer_weights(tensor.elements);
    DeviceBuffer<uint16_t> bf16_input(tensor.spec.cols);
    DeviceBuffer<int64_t> integer_input(tensor.spec.cols);
    DeviceBuffer<float> bf16_output(tensor.spec.rows);
    DeviceBuffer<uint64_t> integer_output(tensor.spec.rows);

    CUDA_CHECK(cudaMemcpy(
        bf16_weights.get(),
        tensor.bf16,
        bf16_weights.bytes(),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        integer_weights.get(),
        tensor.q48,
        integer_weights.bytes(),
        cudaMemcpyHostToDevice
    ));

    std::vector<real_weight::DotReference> timing_integer_reference;
    std::vector<real_weight::FloatingReference> timing_bf16_reference;
    for (std::size_t probe = 0; probe < PROBE_OFFSETS.size(); ++probe) {
        const std::size_t offset = PROBE_OFFSETS[probe];
        if (offset > embedding.elements
            || static_cast<std::size_t>(tensor.spec.cols)
                > embedding.elements - offset) {
            fail("probe slice exceeds embedding tensor");
        }
        const uint16_t *host_bf16_input = embedding.bf16 + offset;
        const int64_t *host_integer_input = embedding.q48 + offset;
        std::size_t probe_rounded = 0;
        for (int column = 0; column < tensor.spec.cols; ++column) {
            int64_t converted = 0;
            if (!real_weight::bf16_to_q48(
                    host_bf16_input[column], &converted
                )
                || converted
                    != bfloat16_to_q1648(host_bf16_input[column])
                || converted != host_integer_input[column]) {
                fail("probe BF16/MGW identity mismatch");
            }
            if (!real_weight::bf16_is_exact_q48(host_bf16_input[column])) {
                ++probe_rounded;
            }
        }
        if (probe_rounded != PROBE_EXPECTED_ROUNDED[probe]) {
            fail("probe repository-conversion rounding census changed");
        }
        std::printf(
            "PROBE_IDENTITY tensor_id=%s probe=%zu offset=%zu elements=%d "
            "rounded_elements=%zu status=PASS\n",
            tensor.spec.id,
            probe,
            offset,
            tensor.spec.cols,
            probe_rounded
        );
        const auto integer_references =
            make_integer_references(tensor, host_integer_input);
        const auto bf16_references =
            make_bf16_references(tensor, host_bf16_input, bf16_oracle);
        if (probe == 0) {
            timing_integer_reference = integer_references;
            timing_bf16_reference = bf16_references;
        }
        require_integer_poison_distinct(
            integer_references, 0xa5, "pre-timing poison A"
        );
        require_integer_poison_distinct(
            integer_references, 0x5a, "pre-timing poison B"
        );
        CUDA_CHECK(cudaMemcpy(
            bf16_input.get(),
            host_bf16_input,
            bf16_input.bytes(),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            integer_input.get(),
            host_integer_input,
            integer_input.bytes(),
            cudaMemcpyHostToDevice
        ));

        const auto integer_first = run_integer_once(
            integer_weights,
            integer_input,
            integer_output,
            tensor.spec.rows,
            tensor.spec.cols,
            0xa5
        );
        const auto integer_second = run_integer_once(
            integer_weights,
            integer_input,
            integer_output,
            tensor.spec.rows,
            tensor.spec.cols,
            0x5a
        );
        require_integer_outputs(
            tensor.spec.id,
            static_cast<int>(probe),
            integer_first,
            integer_references
        );
        if (integer_first != integer_second) {
            fail("integer real-weight execution is nondeterministic");
        }

        const auto bf16_first = run_bf16_once(
            handle,
            bf16_weights,
            bf16_input,
            bf16_output,
            tensor.spec.rows,
            tensor.spec.cols,
            BF16_POISON_A
        );
        const auto bf16_second = run_bf16_once(
            handle,
            bf16_weights,
            bf16_input,
            bf16_output,
            tensor.spec.rows,
            tensor.spec.cols,
            BF16_POISON_B
        );
        require_bf16_outputs(
            tensor.spec.id,
            static_cast<int>(probe),
            tensor.spec.cols,
            bf16_first,
            bf16_references,
            integer_references,
            BF16_POISON_A
        );
        require_bf16_outputs(
            tensor.spec.id,
            static_cast<int>(probe + PROBE_OFFSETS.size()),
            tensor.spec.cols,
            bf16_second,
            bf16_references,
            integer_references,
            BF16_POISON_B
        );
        if (std::memcmp(
                bf16_first.data(),
                bf16_second.data(),
                bf16_first.size() * sizeof(float)
            ) != 0) {
            fail("BF16 real-weight execution is not bit-stable");
        }
    }

    const std::size_t timing_offset = PROBE_OFFSETS[0];
    CUDA_CHECK(cudaMemcpy(
        bf16_input.get(),
        embedding.bf16 + timing_offset,
        bf16_input.bytes(),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        integer_input.get(),
        embedding.q48 + timing_offset,
        integer_input.bytes(),
        cudaMemcpyHostToDevice
    ));

    auto run_regime = [&](const char *name, bool use_pressure) {
        run_timing_regime(
            tensor,
            handle,
            bf16_weights,
            bf16_input,
            bf16_output,
            integer_weights,
            integer_input,
            integer_output,
            pressure,
            timing_integer_reference,
            timing_bf16_reference,
            timing_offset,
            name,
            use_pressure
        );
    };
    const bool pressure_first =
        tensor.spec.id[0] == 'R'
        && tensor.spec.id[1] == 'W'
        && ((tensor.spec.id[2] - '0') & 1) != 0;
    if (pressure_first) {
        run_regime("after_l2_pressure", true);
        run_regime("repeat_hot", false);
    } else {
        run_regime("repeat_hot", false);
        run_regime("after_l2_pressure", true);
    }

    require_integer_poison_distinct(
        timing_integer_reference, 0x3c, "post-timing poison A"
    );
    require_integer_poison_distinct(
        timing_integer_reference, 0xc3, "post-timing poison B"
    );
    const auto final_integer_first = run_integer_once(
        integer_weights,
        integer_input,
        integer_output,
        tensor.spec.rows,
        tensor.spec.cols,
        0x3c
    );
    const auto final_integer_second = run_integer_once(
        integer_weights,
        integer_input,
        integer_output,
        tensor.spec.rows,
        tensor.spec.cols,
        0xc3
    );
    require_integer_outputs(
        tensor.spec.id,
        99,
        final_integer_first,
        timing_integer_reference
    );
    if (final_integer_first != final_integer_second) {
        fail("post-timing integer distinct-poison outputs differ");
    }

    const auto final_bf16_first = run_bf16_once(
        handle,
        bf16_weights,
        bf16_input,
        bf16_output,
        tensor.spec.rows,
        tensor.spec.cols,
        BF16_POISON_A
    );
    const auto final_bf16_second = run_bf16_once(
        handle,
        bf16_weights,
        bf16_input,
        bf16_output,
        tensor.spec.rows,
        tensor.spec.cols,
        BF16_POISON_B
    );
    require_bf16_outputs(
        tensor.spec.id,
        99,
        tensor.spec.cols,
        final_bf16_first,
        timing_bf16_reference,
        timing_integer_reference,
        BF16_POISON_A
    );
    require_bf16_outputs(
        tensor.spec.id,
        100,
        tensor.spec.cols,
        final_bf16_second,
        timing_bf16_reference,
        timing_integer_reference,
        BF16_POISON_B
    );
    if (std::memcmp(
            final_bf16_first.data(),
            final_bf16_second.data(),
            final_bf16_first.size() * sizeof(float)
        ) != 0) {
        fail("post-timing BF16 distinct-poison outputs differ");
    }
    std::printf(
        "POST_TIMING tensor_id=%s probe_offset=%zu "
        "integer_distinct_poison=PASS bf16_distinct_poison=PASS\n",
        tensor.spec.id,
        timing_offset
    );
}

struct Arguments {
    const char *safetensors = nullptr;
    const char *mgw = nullptr;
};

Arguments parse_arguments(int argc, char **argv) {
    if (argc != 5
        || std::strcmp(argv[1], "--safetensors") != 0
        || std::strcmp(argv[3], "--mgw") != 0) {
        fail(
            "exact arguments required: "
            "--safetensors PATH --mgw PATH"
        );
    }
    return {argv[2], argv[4]};
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties {};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        int runtime_version = 0;
        int driver_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        int cublas_version = 0;
        std::printf(
            "ENV source_base=%s gpu_name=%s compute=%d.%d "
            "memory_bytes=%zu l2_bytes=%d cuda_runtime=%d cuda_driver=%d "
            "compiled_cudart=%d compiled_cublas=%d\n",
            PUBLIC_SOURCE_COMMIT,
            properties.name,
            properties.major,
            properties.minor,
            properties.totalGlobalMem,
            properties.l2CacheSize,
            runtime_version,
            driver_version,
            CUDART_VERSION,
            CUBLAS_VERSION
        );
        if (!real_weight::target_environment_matches(
                properties.name,
                properties.major,
                properties.minor,
                properties.totalGlobalMem,
                runtime_version
            )) {
            fail("GPU/runtime does not match the pinned target");
        }

        CublasHandle cublas;
        CUBLAS_CHECK(cublasGetVersion(cublas.get(), &cublas_version));
        std::printf(
            "ENV_DETAIL cublas_runtime=%d target_gate=PASS\n",
            cublas_version
        );
        run_bf16_layout_self_test(cublas.get());
        run_integer_self_test();

        SafetensorModel safetensors(arguments.safetensors);
        MgwModel mgw(arguments.mgw);
        const Bf16OracleTable bf16_oracle;
        std::vector<TensorView> tensors;
        tensors.reserve(std::size(TENSOR_SPECS));
        for (const TensorSpec &spec : TENSOR_SPECS) {
            tensors.push_back(bind_tensor(safetensors, mgw, spec));
        }
        for (const TensorView &tensor : tensors) {
            verify_tensor_identity(tensor);
        }

        const TensorView *embedding = nullptr;
        for (const TensorView &tensor : tensors) {
            if (std::strcmp(tensor.spec.name, EMBEDDING_NAME) == 0) {
                embedding = &tensor;
            }
        }
        if (embedding == nullptr) fail("embedding tensor is not selected");

        std::size_t pressure_bytes = std::max<std::size_t>(
            64ULL * 1024 * 1024,
            static_cast<std::size_t>(properties.l2CacheSize) * 2
        );
        if (pressure_bytes > 512ULL * 1024 * 1024) {
            fail("L2 pressure allocation exceeds fixed ceiling");
        }
        pressure_bytes =
            (pressure_bytes + sizeof(uint64_t) - 1)
            / sizeof(uint64_t) * sizeof(uint64_t);
        DeviceBuffer<uint64_t> pressure(
            pressure_bytes / sizeof(uint64_t)
        );
        CUDA_CHECK(cudaMemset(pressure.get(), 0x35, pressure.bytes()));

        for (const TensorView &tensor : tensors) {
            run_tensor(
                tensor, *embedding, cublas.get(), pressure, bf16_oracle
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cublas.close_checked();
        std::puts(
            "DIMENSIONS provenance=PASS representation_identity=PASS "
            "q48_grid_census=PASS "
            "integer_exactness=PASS floating_error_bound=PASS "
            "timed_output_integrity=PASS"
        );
        std::puts(
            "FINAL disposition=OPEN_NEXT_QUESTION "
            "reason=bounded_evidence_ready_for_interpretation "
            "scope=real_weights_probe_vectors_only "
            "nonclaims=no_layer_no_tokens_no_runtime_no_best_integer_kernel"
        );
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(
            stderr,
            "FINAL disposition=INVALID reason=%s\n",
            error.what()
        );
        return 1;
    }
}
