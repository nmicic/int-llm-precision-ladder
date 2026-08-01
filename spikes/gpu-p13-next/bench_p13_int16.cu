/*
 * P13-next: matched CUDA-core M=1 GEMV with FP32, wide signed-int32
 * weights, and losslessly packed signed-int16 weights.
 *
 * No cuBLAS, WMMA or tensor cores. The primary comparison is the two integer
 * lanes: they consume identical canonical weight codes and int32 activation
 * codes, accumulate in signed int64, and apply one final mathematical floor
 * shift. Only resident weight storage/load width differs.
 *
 * Build:
 *   nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
 *     bench_p13_int16.cu -o bench_p13_int16
 */

#include "p13_int16_math.h"
#include "p13_int16_selftest.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef P13_BENCH_SOURCE_SHA256
#define P13_BENCH_SOURCE_SHA256 "UNLOCKED"
#endif
#ifndef P13_MATH_HEADER_SHA256
#define P13_MATH_HEADER_SHA256 "UNLOCKED"
#endif
#ifndef P13_SELFTEST_HEADER_SHA256
#define P13_SELFTEST_HEADER_SHA256 "UNLOCKED"
#endif
#ifndef P13_CANDIDATE_SHA256
#define P13_CANDIDATE_SHA256 "UNLOCKED"
#endif
#ifndef P13_BUILD_ID
#define P13_BUILD_ID "UNLOCKED"
#endif

namespace {

constexpr unsigned WARP_SIZE = 32;
constexpr unsigned WARPS_PER_BLOCK = 8;
constexpr unsigned BLOCK_THREADS = WARP_SIZE * WARPS_PER_BLOCK;
constexpr unsigned WEIGHT_FRACTION_BITS = 12;
constexpr long double FP32_EPSILON = 0x1p-24L;
constexpr std::size_t MIN_PRESSURE_BYTES =
    256ULL * 1024ULL * 1024ULL;
constexpr int32_t INT_POISON_A = INT32_C(-1515870811);
constexpr int32_t INT_POISON_B = INT32_C(1515870810);
constexpr uint32_t FP_POISON_A = UINT32_C(0x7fc12345);
constexpr uint32_t FP_POISON_B = UINT32_C(0xffc54321);

struct Shape {
    const char *id;
    int outputs;
    int inputs;
};

constexpr Shape SHAPES[] = {
    {"q_o_proj", 2048, 2048},
    {"kv_proj", 256, 2048},
    {"gate_up", 5632, 2048},
    {"down_proj", 2048, 5632},
};

struct EvidenceCounts {
    std::size_t samples = 0;
    std::size_t summaries = 0;
    std::size_t boundaries = 0;
    std::size_t preflights = 0;
    std::size_t posts = 0;
    std::size_t shapes = 0;
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void cuda_check(cudaError_t status, const char *expression, int line) {
    if (status != cudaSuccess) {
        fail(
            std::string("CUDA failure line ")
            + std::to_string(line)
            + " for "
            + expression
            + ": "
            + cudaGetErrorString(status)
        );
    }
}

#define CUDA_CHECK(expression) \
    cuda_check((expression), #expression, __LINE__)

void check_launch(const char *label) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        fail(
            std::string("kernel launch failed for ")
            + label
            + ": "
            + cudaGetErrorString(status)
        );
    }
}

std::size_t checked_elements(int outputs, int inputs) {
    if (outputs <= 0 || inputs <= 0) fail("non-positive shape");
    const std::size_t n = static_cast<std::size_t>(outputs);
    const std::size_t k = static_cast<std::size_t>(inputs);
    if (n > std::numeric_limits<std::size_t>::max() / k) {
        fail("shape element count overflow");
    }
    return n * k;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ == 0
            || count_ > std::numeric_limits<std::size_t>::max()
                / sizeof(T)) {
            fail("invalid device allocation size");
        }
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&data_),
            count_ * sizeof(T)
        ));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get() { return data_; }
    const T *get() const { return data_; }
    std::size_t count() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T *data_ = nullptr;
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
            cuda_check(status, "cudaEventCreate(stop)", __LINE__);
        }
    }

    ~EventPair() {
        if (start_ != nullptr) cudaEventDestroy(start_);
        if (stop_ != nullptr) cudaEventDestroy(stop_);
    }

    cudaEvent_t start() const { return start_; }
    cudaEvent_t stop() const { return stop_; }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

uint64_t splitmix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

int32_t generated_weight(std::size_t row, std::size_t column) {
    if (column == 0) return -16384;
    if (column == 1) return 16383;
    if (column == 2) return -256;
    if (column == 3) return 255;
    const uint64_t mixed = splitmix64(
        static_cast<uint64_t>(row) * UINT64_C(0x100000001b3)
        + static_cast<uint64_t>(column)
    );
    return static_cast<int32_t>(mixed & UINT64_C(0x7fff)) - 16384;
}

int32_t generated_activation(std::size_t column) {
    constexpr int32_t RANGE = 400001;
    if (column == 0) return 131072;
    if (column == 1) return -131071;
    if (column == 2) return 65536;
    if (column == 3) return -65537;
    const uint64_t mixed =
        splitmix64(static_cast<uint64_t>(column) + UINT64_C(0x243f6a88));
    return static_cast<int32_t>(mixed % RANGE) - 200000;
}

struct FloatingReference {
    long double exact;
    long double bound;
};

struct Fixture {
    Shape shape{};
    std::vector<int32_t> weights_wide;
    std::vector<int16_t> weights_packed;
    std::vector<float> weights_fp32;
    std::vector<int32_t> activations_i32;
    std::vector<float> activations_fp32;
    std::vector<int32_t> expected_i32;
    std::vector<FloatingReference> expected_fp32;
    double pack_microseconds = 0.0;
    uint64_t maximum_absolute_product_sum = 0;
    uint64_t maximum_absolute_exact_sum = 0;
};

long double fp32_bound(std::size_t count, long double absolute_sum) {
    if (count == 0
        || count > std::numeric_limits<std::size_t>::max()
            - (WARP_SIZE - 1)
        || !std::isfinite(absolute_sum)
        || absolute_sum < 0.0L) {
        fail("invalid FP32 error-bound input");
    }
    const std::size_t lane_terms =
        (count + WARP_SIZE - 1) / WARP_SIZE;
    if (lane_terms
        > std::numeric_limits<std::size_t>::max() - 5) {
        fail("FP32 error-bound operation count overflow");
    }
    const long double operations =
        static_cast<long double>(lane_terms + 5);
    const long double product = operations * FP32_EPSILON;
    if (!std::isfinite(product)
        || product < 0.0L
        || product >= 0.5L) {
        fail("invalid FP32 error-bound domain");
    }
    const long double gamma = product / (1.0L - product);
    const long double bound =
        8.0L * gamma * absolute_sum + 0x1p-20L;
    if (!std::isfinite(bound) || bound < 0.0L) {
        fail("non-finite FP32 error bound");
    }
    return bound;
}

Fixture make_fixture(const Shape &shape) {
    Fixture fixture;
    fixture.shape = shape;
    const std::size_t elements =
        checked_elements(shape.outputs, shape.inputs);
    fixture.weights_wide.resize(elements);
    fixture.weights_packed.resize(elements);
    fixture.weights_fp32.resize(elements);
    fixture.activations_i32.resize(static_cast<std::size_t>(shape.inputs));
    fixture.activations_fp32.resize(static_cast<std::size_t>(shape.inputs));
    fixture.expected_i32.resize(static_cast<std::size_t>(shape.outputs));
    fixture.expected_fp32.resize(static_cast<std::size_t>(shape.outputs));

    for (int k = 0; k < shape.inputs; ++k) {
        const int32_t code =
            generated_activation(static_cast<std::size_t>(k));
        fixture.activations_i32[static_cast<std::size_t>(k)] = code;
        fixture.activations_fp32[static_cast<std::size_t>(k)] =
            std::ldexp(static_cast<float>(code), -16);
        if (!std::isfinite(
                fixture.activations_fp32[static_cast<std::size_t>(k)]
            )
            || std::ldexp(
                fixture.activations_fp32[static_cast<std::size_t>(k)],
                16
            ) != static_cast<float>(code)) {
            fail("activation is not exactly representable in FP32 control");
        }
    }
    if (fixture.activations_i32[0]
        <= std::numeric_limits<int16_t>::max()) {
        fail("fixture failed to preserve an int32-only activation");
    }

    for (int row = 0; row < shape.outputs; ++row) {
        const std::size_t row_offset =
            static_cast<std::size_t>(row)
            * static_cast<std::size_t>(shape.inputs);
        for (int k = 0; k < shape.inputs; ++k) {
            const std::size_t index =
                row_offset + static_cast<std::size_t>(k);
            const int32_t code = generated_weight(
                static_cast<std::size_t>(row),
                static_cast<std::size_t>(k)
            );
            fixture.weights_wide[index] = code;
            fixture.weights_fp32[index] =
                std::ldexp(static_cast<float>(code), -12);
            if (!std::isfinite(fixture.weights_fp32[index])
                || std::ldexp(fixture.weights_fp32[index], 12)
                    != static_cast<float>(code)) {
                fail("weight is not exactly representable in FP32 control");
            }
        }
    }

    const auto pack_start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < elements; ++index) {
        if (!p13_int16::pack_weight(
                fixture.weights_wide[index],
                15,
                &fixture.weights_packed[index]
            )) {
            fail("lossless int16 packing rejected generated fixture");
        }
    }
    const auto pack_stop = std::chrono::steady_clock::now();
    fixture.pack_microseconds =
        std::chrono::duration<double, std::micro>(
            pack_stop - pack_start
        ).count();

    for (std::size_t index = 0; index < elements; ++index) {
        if (p13_int16::decode_weight(fixture.weights_packed[index])
            != fixture.weights_wide[index]) {
            fail("full fixture pack/decode identity failed");
        }
    }

    for (int row = 0; row < shape.outputs; ++row) {
        const std::size_t row_offset =
            static_cast<std::size_t>(row)
            * static_cast<std::size_t>(shape.inputs);
        p13_int16::DotReference reference{};
        if (!p13_int16::dot_reference(
                fixture.weights_wide.data() + row_offset,
                fixture.activations_i32.data(),
                static_cast<std::size_t>(shape.inputs),
                WEIGHT_FRACTION_BITS,
                &reference
            )) {
            fail("signed-128 integer reference rejected fixture");
        }
        p13_int16::DotReference packed_reference{};
        if (!p13_int16::dot_reference_packed(
                fixture.weights_packed.data() + row_offset,
                fixture.activations_i32.data(),
                static_cast<std::size_t>(shape.inputs),
                WEIGHT_FRACTION_BITS,
                &packed_reference
            )
            || packed_reference.exact_sum != reference.exact_sum
            || packed_reference.output != reference.output) {
            fail("packed and wide host references differ");
        }
        fixture.expected_i32[static_cast<std::size_t>(row)] =
            reference.output;
        fixture.maximum_absolute_product_sum = std::max(
            fixture.maximum_absolute_product_sum,
            reference.absolute_product_sum
        );
        const uint64_t absolute_exact_sum = static_cast<uint64_t>(
            reference.exact_sum < 0
                ? -reference.exact_sum
                : reference.exact_sum
        );
        fixture.maximum_absolute_exact_sum = std::max(
            fixture.maximum_absolute_exact_sum,
            absolute_exact_sum
        );

        long double exact = 0.0L;
        long double absolute_sum = 0.0L;
        for (int k = 0; k < shape.inputs; ++k) {
            const std::size_t index =
                row_offset + static_cast<std::size_t>(k);
            const long double product =
                static_cast<long double>(fixture.weights_fp32[index])
                * static_cast<long double>(
                    fixture.activations_fp32[static_cast<std::size_t>(k)]
                );
            exact += product;
            absolute_sum += std::fabs(product);
        }
        fixture.expected_fp32[static_cast<std::size_t>(row)] = {
            exact,
            fp32_bound(
                static_cast<std::size_t>(shape.inputs),
                absolute_sum
            ),
        };
    }
    return fixture;
}

__device__ bool floor_shift_i64_to_i32(
    int64_t value,
    unsigned bits,
    int32_t *output
) {
    if (output == nullptr || bits != WEIGHT_FRACTION_BITS) return false;
    const uint64_t scale = UINT64_C(1) << bits;
    if (value >= 0) {
        const uint64_t quotient =
            static_cast<uint64_t>(value) >> bits;
        if (quotient > static_cast<uint64_t>(INT32_MAX)) {
            return false;
        }
        *output = static_cast<int32_t>(quotient);
        return true;
    }
    const uint64_t value_magnitude =
        static_cast<uint64_t>(-(value + 1)) + UINT64_C(1);
    uint64_t quotient = value_magnitude >> bits;
    if ((value_magnitude & (scale - 1)) != 0) ++quotient;
    if (quotient > (UINT64_C(1) << 31)) return false;
    if (quotient == (UINT64_C(1) << 31)) {
        *output = INT32_MIN;
        return true;
    }
    *output = -static_cast<int32_t>(quotient);
    return true;
}

template <typename Weight>
__global__ void gemv_integer_warp(
    int32_t *output,
    const Weight *weights,
    const int32_t *activation,
    int outputs,
    int inputs,
    unsigned weight_fraction_bits
) {
    const unsigned warp_in_block = threadIdx.x / WARP_SIZE;
    const unsigned lane = threadIdx.x % WARP_SIZE;
    const int row =
        static_cast<int>(blockIdx.x * WARPS_PER_BLOCK + warp_in_block);
    if (row >= outputs) return;

    const Weight *row_weights =
        weights + static_cast<int64_t>(row) * inputs;
    int64_t sum = 0;
    for (int k = static_cast<int>(lane); k < inputs; k += WARP_SIZE) {
        sum += static_cast<int64_t>(row_weights[k])
            * static_cast<int64_t>(activation[k]);
    }
    for (unsigned offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(UINT32_MAX, sum, offset);
    }
    if (lane == 0) {
        int32_t shifted = INT32_C(-1515870811);
        if (!floor_shift_i64_to_i32(
                sum, weight_fraction_bits, &shifted
            )) {
            shifted = INT32_C(-1515870811);
        }
        output[row] = shifted;
    }
}

__global__ void gemv_integer_zero_extend_negative(
    int32_t *output,
    const uint16_t *weights,
    const int32_t *activation,
    int outputs,
    int inputs,
    unsigned weight_fraction_bits
) {
    const unsigned warp_in_block = threadIdx.x / WARP_SIZE;
    const unsigned lane = threadIdx.x % WARP_SIZE;
    const int row =
        static_cast<int>(blockIdx.x * WARPS_PER_BLOCK + warp_in_block);
    if (row >= outputs) return;
    const uint16_t *row_weights =
        weights + static_cast<int64_t>(row) * inputs;
    int64_t sum = 0;
    for (int k = static_cast<int>(lane); k < inputs; k += WARP_SIZE) {
        sum += static_cast<int64_t>(row_weights[k])
            * static_cast<int64_t>(activation[k]);
    }
    for (unsigned offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(UINT32_MAX, sum, offset);
    }
    if (lane == 0) {
        int32_t shifted = INT32_C(-1515870811);
        if (!floor_shift_i64_to_i32(
                sum, weight_fraction_bits, &shifted
            )) {
            shifted = INT32_C(-1515870811);
        }
        output[row] = shifted;
    }
}

__global__ void floor_shift_selftest_kernel(
    const int64_t *values,
    int32_t *outputs,
    uint8_t *accepted,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int32_t shifted = 0;
    const bool ok = floor_shift_i64_to_i32(
        values[index], WEIGHT_FRACTION_BITS, &shifted
    );
    accepted[index] = ok ? UINT8_C(1) : UINT8_C(0);
    outputs[index] = ok ? shifted : INT32_C(0x12345678);
}

__global__ void gemv_fp32_warp(
    float *output,
    const float *weights,
    const float *activation,
    int outputs,
    int inputs
) {
    const unsigned warp_in_block = threadIdx.x / WARP_SIZE;
    const unsigned lane = threadIdx.x % WARP_SIZE;
    const int row =
        static_cast<int>(blockIdx.x * WARPS_PER_BLOCK + warp_in_block);
    if (row >= outputs) return;

    const float *row_weights =
        weights + static_cast<int64_t>(row) * inputs;
    float sum = 0.0f;
    for (int k = static_cast<int>(lane); k < inputs; k += WARP_SIZE) {
        sum = fmaf(row_weights[k], activation[k], sum);
    }
    for (unsigned offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(UINT32_MAX, sum, offset);
    }
    if (lane == 0) output[row] = sum;
}

__global__ void fill_i32(int32_t *output, std::size_t count, int32_t value) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = value;
}

__global__ void fill_fp32_bits(
    float *output,
    std::size_t count,
    uint32_t bits
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __uint_as_float(bits);
}

__global__ void pressure_kernel(uint64_t *data, std::size_t count) {
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = first; index < count; index += stride) {
        data[index] =
            data[index] * UINT64_C(6364136223846793005)
            + static_cast<uint64_t>(index)
            + UINT64_C(1442695040888963407);
    }
}

unsigned grid_for_rows(int outputs) {
    return (
        static_cast<unsigned>(outputs)
        + WARPS_PER_BLOCK
        - 1
    ) / WARPS_PER_BLOCK;
}

unsigned grid_for_count(std::size_t count) {
    constexpr std::size_t threads = 256;
    const std::size_t blocks = (count + threads - 1) / threads;
    if (blocks == 0 || blocks > std::numeric_limits<unsigned>::max()) {
        fail("fill grid size invalid");
    }
    return static_cast<unsigned>(blocks);
}

void launch_wide(
    int32_t *output,
    const int32_t *weights,
    const int32_t *activation,
    int outputs,
    int inputs
) {
    gemv_integer_warp<<<grid_for_rows(outputs), BLOCK_THREADS>>>(
        output,
        weights,
        activation,
        outputs,
        inputs,
        WEIGHT_FRACTION_BITS
    );
    check_launch("wide integer GEMV");
}

void launch_packed(
    int32_t *output,
    const int16_t *weights,
    const int32_t *activation,
    int outputs,
    int inputs
) {
    gemv_integer_warp<<<grid_for_rows(outputs), BLOCK_THREADS>>>(
        output,
        weights,
        activation,
        outputs,
        inputs,
        WEIGHT_FRACTION_BITS
    );
    check_launch("packed integer GEMV");
}

void launch_fp32(
    float *output,
    const float *weights,
    const float *activation,
    int outputs,
    int inputs
) {
    gemv_fp32_warp<<<grid_for_rows(outputs), BLOCK_THREADS>>>(
        output, weights, activation, outputs, inputs
    );
    check_launch("FP32 GEMV");
}

void apply_pressure(DeviceBuffer<uint64_t> &pressure) {
    pressure_kernel<<<4096, 256>>>(pressure.get(), pressure.count());
    check_launch("L2 pressure");
}

void poison_i32(
    DeviceBuffer<int32_t> &output,
    int32_t value
) {
    fill_i32<<<grid_for_count(output.count()), 256>>>(
        output.get(), output.count(), value
    );
    check_launch("integer poison");
}

void poison_fp32(
    DeviceBuffer<float> &output,
    uint32_t bits
) {
    fill_fp32_bits<<<grid_for_count(output.count()), 256>>>(
        output.get(), output.count(), bits
    );
    check_launch("FP32 poison");
}

struct FloatingVerdict {
    std::size_t calls = 0;
    std::size_t elements = 0;
    std::size_t violations = 0;
    std::size_t invalid_reference = 0;
    std::size_t non_finite = 0;
    std::size_t poison_hits = 0;
    std::size_t first_violation = 0;
    long double maximum_error = 0.0L;
    long double maximum_bound = 0.0L;

    bool passed() const {
        return calls != 0
            && elements != 0
            && violations == 0
            && invalid_reference == 0
            && non_finite == 0
            && poison_hits == 0;
    }

    void merge(const FloatingVerdict &other) {
        if (violations == 0 && other.violations != 0) {
            first_violation = other.first_violation;
        }
        calls += other.calls;
        elements += other.elements;
        violations += other.violations;
        invalid_reference += other.invalid_reference;
        non_finite += other.non_finite;
        poison_hits += other.poison_hits;
        maximum_error = std::max(
            maximum_error, other.maximum_error
        );
        maximum_bound = std::max(
            maximum_bound, other.maximum_bound
        );
    }
};

FloatingVerdict verify_fp32(
    const std::vector<float> &actual,
    const std::vector<FloatingReference> &expected,
    uint32_t poison_bits
) {
    if (actual.empty() || actual.size() != expected.size()) {
        fail("invalid FP32 verification buffers");
    }
    FloatingVerdict verdict;
    verdict.calls = 1;
    verdict.elements = actual.size();
    verdict.first_violation = actual.size();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        bool element_violation = false;
        uint32_t bits = 0;
        std::memcpy(&bits, &actual[index], sizeof(bits));
        if (bits == poison_bits) {
            ++verdict.poison_hits;
            element_violation = true;
        }
        if (!std::isfinite(expected[index].exact)
            || !std::isfinite(expected[index].bound)
            || expected[index].bound < 0.0L) {
            ++verdict.invalid_reference;
            element_violation = true;
        }
        if (!std::isfinite(actual[index])) {
            ++verdict.non_finite;
            element_violation = true;
        }
        if (!element_violation) {
            const long double error = std::fabs(
                static_cast<long double>(actual[index])
                - expected[index].exact
            );
            if (!std::isfinite(error)) {
                ++verdict.non_finite;
                element_violation = true;
            } else {
                verdict.maximum_error = std::max(
                    verdict.maximum_error, error
                );
                verdict.maximum_bound = std::max(
                    verdict.maximum_bound, expected[index].bound
                );
                if (error > expected[index].bound) {
                    element_violation = true;
                }
            }
        }
        if (element_violation) {
            if (verdict.violations == 0) {
                verdict.first_violation = index;
            }
            ++verdict.violations;
        }
    }
    return verdict;
}

void require_integer_poison_distinct(
    const std::vector<int32_t> &expected,
    int32_t poison
) {
    if (std::find(expected.begin(), expected.end(), poison)
        != expected.end()) {
        fail("integer poison collides with expected output");
    }
}

void verify_integer_output(
    DeviceBuffer<int32_t> &device,
    const std::vector<int32_t> &expected,
    std::vector<int32_t> &host
) {
    if (host.size() != expected.size()
        || device.count() != expected.size()) {
        fail("invalid integer verification buffers");
    }
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        device.get(),
        device.bytes(),
        cudaMemcpyDeviceToHost
    ));
    const auto verdict = p13_int16::verify_i32(
        host.data(), expected.data(), expected.size()
    );
    if (!verdict.passed()) {
        fail(
            std::string("integer output mismatch count=")
            + std::to_string(verdict.mismatches)
            + " first="
            + std::to_string(verdict.first_mismatch)
        );
    }
}

void verify_fp32_output(
    DeviceBuffer<float> &device,
    const std::vector<FloatingReference> &expected,
    std::vector<float> &host,
    uint32_t poison_bits,
    FloatingVerdict *observed
) {
    if (observed == nullptr
        || host.size() != expected.size()
        || device.count() != expected.size()) {
        fail("invalid FP32 output verification state");
    }
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        device.get(),
        device.bytes(),
        cudaMemcpyDeviceToHost
    ));
    const FloatingVerdict verdict =
        verify_fp32(host, expected, poison_bits);
    if (!verdict.passed()) {
        fail(
            std::string("FP32 verification failed violations=")
            + std::to_string(verdict.violations)
            + " invalid_reference="
            + std::to_string(verdict.invalid_reference)
            + " poison_hits="
            + std::to_string(verdict.poison_hits)
        );
    }
    observed->merge(verdict);
}

enum class Lane {
    FP32,
    WIDE,
    PACKED,
};

const char *lane_name(Lane lane) {
    switch (lane) {
        case Lane::FP32: return "fp32_control";
        case Lane::WIDE: return "wide_int32_control";
        case Lane::PACKED: return "packed_int16_candidate";
    }
    fail("unknown lane");
}

std::size_t lane_slot(Lane lane) {
    switch (lane) {
        case Lane::FP32: return 0;
        case Lane::WIDE: return 1;
        case Lane::PACKED: return 2;
    }
    fail("unknown lane slot");
}

struct DeviceFixture {
    explicit DeviceFixture(const Fixture &fixture)
        : weights_wide(fixture.weights_wide.size()),
          weights_packed(fixture.weights_packed.size()),
          weights_fp32(fixture.weights_fp32.size()),
          activations_i32(fixture.activations_i32.size()),
          activations_fp32(fixture.activations_fp32.size()),
          output_i32(fixture.expected_i32.size()),
          output_fp32(fixture.expected_fp32.size()) {
        CUDA_CHECK(cudaMemcpy(
            activations_i32.get(),
            fixture.activations_i32.data(),
            activations_i32.bytes(),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            activations_fp32.get(),
            fixture.activations_fp32.data(),
            activations_fp32.bytes(),
            cudaMemcpyHostToDevice
        ));
    }

    DeviceBuffer<int32_t> weights_wide;
    DeviceBuffer<int16_t> weights_packed;
    DeviceBuffer<float> weights_fp32;
    DeviceBuffer<int32_t> activations_i32;
    DeviceBuffer<float> activations_fp32;
    DeviceBuffer<int32_t> output_i32;
    DeviceBuffer<float> output_fp32;
};

double copy_h2d_microseconds(
    void *destination,
    const void *source,
    std::size_t bytes
) {
    const auto start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(
        destination, source, bytes, cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    const double microseconds =
        std::chrono::duration<double, std::micro>(
            stop - start
        ).count();
    if (!std::isfinite(microseconds) || microseconds <= 0.0) {
        fail("invalid diagnostic H2D duration");
    }
    return microseconds;
}

template <typename Callable>
void require_failure(Callable callable, const char *label) {
    bool rejected = false;
    try {
        callable();
    } catch (const std::exception &) {
        rejected = true;
    }
    if (!rejected) {
        fail(std::string("negative control did not reject: ") + label);
    }
}

void run_host_controls() {
    const p13_int16::HostSelfTestResult math =
        p13_int16::run_host_math_selftests();
    if (math.checks != 85) {
        fail("mandatory host math suite check count changed");
    }

    const std::vector<float> exact_actual{1.0f};
    const std::vector<FloatingReference> exact_expected{
        {1.0L, 0.0L}
    };
    const FloatingVerdict exact = verify_fp32(
        exact_actual, exact_expected, FP_POISON_A
    );
    if (!exact.passed()
        || exact.calls != 1
        || exact.elements != 1) {
        fail("valid FP32 verifier control failed");
    }

    std::size_t fp32_negative_cases = 0;
    auto require_fp32_reject = [&](
        const std::vector<float> &actual,
        const std::vector<FloatingReference> &expected,
        uint32_t poison,
        const char *label
    ) {
        ++fp32_negative_cases;
        const FloatingVerdict verdict =
            verify_fp32(actual, expected, poison);
        if (verdict.passed()) {
            fail(std::string("FP32 verifier accepted ") + label);
        }
    };

    require_fp32_reject(
        {2.0f}, exact_expected, FP_POISON_A, "corrupted value"
    );
    require_fp32_reject(
        exact_actual,
        {{std::numeric_limits<long double>::quiet_NaN(), 0.0L}},
        FP_POISON_A,
        "NaN exact reference"
    );
    require_fp32_reject(
        exact_actual,
        {{1.0L, std::numeric_limits<long double>::quiet_NaN()}},
        FP_POISON_A,
        "NaN reference bound"
    );
    require_fp32_reject(
        exact_actual,
        {{std::numeric_limits<long double>::infinity(), 0.0L}},
        FP_POISON_A,
        "infinite exact reference"
    );
    require_fp32_reject(
        exact_actual,
        {{1.0L, std::numeric_limits<long double>::infinity()}},
        FP_POISON_A,
        "infinite reference bound"
    );
    require_fp32_reject(
        exact_actual,
        {{1.0L, -1.0L}},
        FP_POISON_A,
        "negative reference bound"
    );
    require_fp32_reject(
        {std::numeric_limits<float>::infinity()},
        exact_expected,
        FP_POISON_A,
        "infinite device value"
    );
    float poison_value = 0.0f;
    std::memcpy(
        &poison_value, &FP_POISON_A, sizeof(poison_value)
    );
    require_fp32_reject(
        {poison_value},
        {{0.0L, std::numeric_limits<long double>::max()}},
        FP_POISON_A,
        "preserved poison"
    );

    FloatingVerdict aggregate;
    if (aggregate.passed()) {
        fail("empty FP32 aggregate passed");
    }
    aggregate.merge(exact);
    if (!aggregate.passed()
        || aggregate.calls != 1
        || aggregate.elements != 1) {
        fail("valid FP32 aggregate failed");
    }
    aggregate.merge(verify_fp32(
        {2.0f}, exact_expected, FP_POISON_A
    ));
    if (aggregate.passed()) {
        fail("failed local FP32 verdict did not fail aggregate");
    }

    if (!std::isfinite(fp32_bound(1, 0.0L))) {
        fail("valid FP32 bound control failed");
    }
    require_failure(
        []() { (void)fp32_bound(0, 0.0L); },
        "zero-count FP32 bound"
    );
    require_failure(
        []() { (void)fp32_bound(1, -1.0L); },
        "negative-sum FP32 bound"
    );
    require_failure(
        []() {
            (void)fp32_bound(
                1, std::numeric_limits<long double>::infinity()
            );
        },
        "infinite-sum FP32 bound"
    );

    std::printf(
        "SELFTEST host_gates=PASS math_checks=%zu "
        "fp32_negative_cases=%zu aggregate_negative=PASS "
        "bound_negative=PASS\n",
        math.checks,
        fp32_negative_cases
    );
}

void upload_integer_weights(
    const Fixture &fixture,
    DeviceFixture &device
) {
    CUDA_CHECK(cudaMemcpy(
        device.weights_wide.get(),
        fixture.weights_wide.data(),
        device.weights_wide.bytes(),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        device.weights_packed.get(),
        fixture.weights_packed.data(),
        device.weights_packed.bytes(),
        cudaMemcpyHostToDevice
    ));
}

bool integer_output_matches(
    DeviceBuffer<int32_t> &device,
    const std::vector<int32_t> &expected,
    std::vector<int32_t> &host
) {
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        device.get(),
        device.bytes(),
        cudaMemcpyDeviceToHost
    ));
    return p13_int16::verify_i32(
        host.data(), expected.data(), expected.size()
    ).passed();
}

Fixture make_literal_fixture() {
    Fixture fixture;
    fixture.shape = {"literal_signed_fixture", 9, 33};
    const std::size_t elements = checked_elements(
        fixture.shape.outputs, fixture.shape.inputs
    );
    fixture.weights_wide.assign(elements, 0);
    fixture.weights_packed.assign(elements, 0);
    fixture.weights_fp32.assign(elements, 0.0f);
    fixture.activations_i32.assign(
        static_cast<std::size_t>(fixture.shape.inputs), 0
    );
    fixture.activations_fp32.assign(
        static_cast<std::size_t>(fixture.shape.inputs), 0.0f
    );
    fixture.expected_i32 = {
        0, -1, -1, -2, -32, -33, -1, 1, 8191
    };
    fixture.expected_fp32.assign(
        static_cast<std::size_t>(fixture.shape.outputs),
        {0.0L, 0.0L}
    );

    fixture.activations_i32[0] = 131072;
    fixture.activations_i32[1] = 131072;
    fixture.activations_i32[2] = 65536;
    fixture.activations_i32[3] = 65537;
    fixture.activations_i32[4] = 4096;
    fixture.activations_i32[5] = 4097;
    fixture.activations_i32[6] = -131073;
    fixture.activations_i32[7] =
        std::numeric_limits<int32_t>::max();
    fixture.activations_i32[8] =
        std::numeric_limits<int32_t>::min();
    fixture.activations_i32[32] = -131071;
    for (int k = 0; k < fixture.shape.inputs; ++k) {
        fixture.activations_fp32[static_cast<std::size_t>(k)] =
            std::ldexp(
                static_cast<float>(
                    fixture.activations_i32[static_cast<std::size_t>(k)]
                ),
                -16
            );
    }

    auto set_weight = [&](int row, int column, int32_t value) {
        const std::size_t index =
            static_cast<std::size_t>(row)
            * static_cast<std::size_t>(fixture.shape.inputs)
            + static_cast<std::size_t>(column);
        fixture.weights_wide[index] = value;
        fixture.weights_fp32[index] =
            std::ldexp(static_cast<float>(value), -12);
    };
    set_weight(0, 0, 16383);
    set_weight(0, 1, -16383);
    set_weight(1, 2, 1);
    set_weight(1, 3, -1);
    set_weight(2, 4, -1);
    set_weight(3, 5, -1);
    set_weight(4, 0, -16384);
    set_weight(4, 1, 16383);
    set_weight(5, 6, 1);
    set_weight(6, 7, 1);
    set_weight(6, 8, 1);
    set_weight(7, 5, 1);
    set_weight(8, 32, -256);

    const std::array<int64_t, 9> exact_sums{{
        INT64_C(0),
        INT64_C(-1),
        INT64_C(-4096),
        INT64_C(-4097),
        INT64_C(-131072),
        INT64_C(-131073),
        INT64_C(-1),
        INT64_C(4097),
        INT64_C(33554176),
    }};

    for (std::size_t index = 0; index < elements; ++index) {
        if (!p13_int16::pack_weight(
                fixture.weights_wide[index],
                15,
                &fixture.weights_packed[index]
            )) {
            fail("literal fixture packing failed");
        }
    }
    for (int row = 0; row < fixture.shape.outputs; ++row) {
        const std::size_t offset =
            static_cast<std::size_t>(row)
            * static_cast<std::size_t>(fixture.shape.inputs);
        p13_int16::DotReference reference{};
        p13_int16::DotReference packed_reference{};
        if (!p13_int16::dot_reference(
                fixture.weights_wide.data() + offset,
                fixture.activations_i32.data(),
                static_cast<std::size_t>(fixture.shape.inputs),
                WEIGHT_FRACTION_BITS,
                &reference
            )
            || !p13_int16::dot_reference_packed(
                fixture.weights_packed.data() + offset,
                fixture.activations_i32.data(),
                static_cast<std::size_t>(fixture.shape.inputs),
                WEIGHT_FRACTION_BITS,
                &packed_reference
            )
            || reference.exact_sum
                != exact_sums[static_cast<std::size_t>(row)]
            || packed_reference.exact_sum != reference.exact_sum
            || reference.output
                != fixture.expected_i32[static_cast<std::size_t>(row)]
            || packed_reference.output != reference.output) {
            fail("literal signed-128 answer disagrees with hard-coded result");
        }
        fixture.maximum_absolute_product_sum = std::max(
            fixture.maximum_absolute_product_sum,
            reference.absolute_product_sum
        );
        fixture.maximum_absolute_exact_sum = std::max(
            fixture.maximum_absolute_exact_sum,
            static_cast<uint64_t>(
                reference.exact_sum < 0
                    ? -reference.exact_sum
                    : reference.exact_sum
            )
        );
    }
    return fixture;
}

void verify_integer_fixture_on_device(const Fixture &fixture) {
    DeviceFixture device(fixture);
    upload_integer_weights(fixture, device);
    require_integer_poison_distinct(
        fixture.expected_i32, INT_POISON_A
    );
    require_integer_poison_distinct(
        fixture.expected_i32, INT_POISON_B
    );
    std::vector<int32_t> host(fixture.expected_i32.size());

    poison_i32(device.output_i32, INT_POISON_A);
    launch_wide(
        device.output_i32.get(),
        device.weights_wide.get(),
        device.activations_i32.get(),
        fixture.shape.outputs,
        fixture.shape.inputs
    );
    verify_integer_output(
        device.output_i32, fixture.expected_i32, host
    );

    poison_i32(device.output_i32, INT_POISON_B);
    launch_packed(
        device.output_i32.get(),
        device.weights_packed.get(),
        device.activations_i32.get(),
        fixture.shape.outputs,
        fixture.shape.inputs
    );
    verify_integer_output(
        device.output_i32, fixture.expected_i32, host
    );
}

void run_floor_device_selftest() {
    const int64_t scale = INT64_C(1) << WEIGHT_FRACTION_BITS;
    const std::vector<int64_t> values{
        0,
        1,
        4095,
        4096,
        4097,
        -1,
        -4095,
        -4096,
        -4097,
        static_cast<int64_t>(std::numeric_limits<int32_t>::max())
            * scale + scale - 1,
        static_cast<int64_t>(std::numeric_limits<int32_t>::max())
            * scale + scale,
        static_cast<int64_t>(std::numeric_limits<int32_t>::min())
            * scale,
        static_cast<int64_t>(std::numeric_limits<int32_t>::min())
            * scale - 1,
        std::numeric_limits<int64_t>::min(),
        std::numeric_limits<int64_t>::max(),
    };
    const std::vector<int32_t> expected{
        0, 0, 0, 1, 1, -1, -1, -1, -2,
        std::numeric_limits<int32_t>::max(),
        INT32_C(0x12345678),
        std::numeric_limits<int32_t>::min(),
        INT32_C(0x12345678),
        INT32_C(0x12345678),
        INT32_C(0x12345678),
    };
    const std::vector<uint8_t> accepted{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0
    };
    DeviceBuffer<int64_t> device_values(values.size());
    DeviceBuffer<int32_t> device_outputs(values.size());
    DeviceBuffer<uint8_t> device_accepted(values.size());
    CUDA_CHECK(cudaMemcpy(
        device_values.get(),
        values.data(),
        device_values.bytes(),
        cudaMemcpyHostToDevice
    ));
    floor_shift_selftest_kernel<<<
        grid_for_count(values.size()), 256
    >>>(
        device_values.get(),
        device_outputs.get(),
        device_accepted.get(),
        values.size()
    );
    check_launch("device floor selftest");
    std::vector<int32_t> observed_outputs(values.size());
    std::vector<uint8_t> observed_accepted(values.size());
    CUDA_CHECK(cudaMemcpy(
        observed_outputs.data(),
        device_outputs.get(),
        device_outputs.bytes(),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        observed_accepted.data(),
        device_accepted.get(),
        device_accepted.bytes(),
        cudaMemcpyDeviceToHost
    ));
    if (observed_outputs != expected
        || observed_accepted != accepted) {
        fail("device floor endpoint selftest failed");
    }
}

void run_device_selftests() {
    constexpr Shape BOUNDARY_SHAPES[] = {
        {"boundary_n1_k1", 1, 1},
        {"boundary_n7_k31", 7, 31},
        {"boundary_n8_k32", 8, 32},
        {"boundary_n9_k33", 9, 33},
        {"boundary_n7_k255", 7, 255},
        {"boundary_n8_k256", 8, 256},
        {"boundary_n9_k257", 9, 257},
        {"boundary_n33_k257", 33, 257},
    };
    for (const Shape &shape : BOUNDARY_SHAPES) {
        verify_integer_fixture_on_device(make_fixture(shape));
    }
    verify_integer_fixture_on_device(make_literal_fixture());
    run_floor_device_selftest();

    const Shape negative_shape{"negative_control_fixture", 33, 257};
    const Fixture fixture = make_fixture(negative_shape);
    DeviceFixture device(fixture);
    upload_integer_weights(fixture, device);
    CUDA_CHECK(cudaMemcpy(
        device.weights_fp32.get(),
        fixture.weights_fp32.data(),
        device.weights_fp32.bytes(),
        cudaMemcpyHostToDevice
    ));
    require_integer_poison_distinct(
        fixture.expected_i32, INT_POISON_A
    );
    require_integer_poison_distinct(
        fixture.expected_i32, INT_POISON_B
    );
    std::vector<int32_t> host_i32(fixture.expected_i32.size());
    std::vector<float> host_fp32(fixture.expected_fp32.size());

    for (const int32_t poison : {INT_POISON_A, INT_POISON_B}) {
        poison_i32(device.output_i32, poison);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (integer_output_matches(
                device.output_i32, fixture.expected_i32, host_i32
            )) {
            fail("integer verifier accepted skipped poisoned output");
        }
    }

    poison_i32(device.output_i32, INT_POISON_A);
    launch_wide(
        device.output_i32.get(),
        device.weights_wide.get(),
        device.activations_i32.get(),
        negative_shape.outputs - 1,
        negative_shape.inputs
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    if (integer_output_matches(
            device.output_i32, fixture.expected_i32, host_i32
        )) {
        fail("wide verifier accepted partial poisoned output");
    }

    poison_i32(device.output_i32, INT_POISON_B);
    launch_packed(
        device.output_i32.get(),
        device.weights_packed.get(),
        device.activations_i32.get(),
        negative_shape.outputs - 1,
        negative_shape.inputs
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    if (integer_output_matches(
            device.output_i32, fixture.expected_i32, host_i32
        )) {
        fail("packed verifier accepted partial poisoned output");
    }

    for (const uint32_t poison : {FP_POISON_A, FP_POISON_B}) {
        poison_fp32(device.output_fp32, poison);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            host_fp32.data(),
            device.output_fp32.get(),
            device.output_fp32.bytes(),
            cudaMemcpyDeviceToHost
        ));
        for (const float value : host_fp32) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            if (bits != poison) {
                fail("FP32 poison bit-pattern selftest failed");
            }
        }
        if (verify_fp32(
                host_fp32, fixture.expected_fp32, poison
            ).passed()) {
            fail("FP32 verifier accepted skipped poisoned output");
        }
    }

    poison_fp32(device.output_fp32, FP_POISON_B);
    gemv_fp32_warp<<<
        grid_for_rows(negative_shape.outputs), BLOCK_THREADS
    >>>(
        device.output_fp32.get(),
        device.weights_fp32.get(),
        device.activations_fp32.get(),
        negative_shape.outputs - 1,
        negative_shape.inputs
    );
    check_launch("partial FP32 negative control");
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(
        host_fp32.data(),
        device.output_fp32.get(),
        device.output_fp32.bytes(),
        cudaMemcpyDeviceToHost
    ));
    if (verify_fp32(
            host_fp32, fixture.expected_fp32, FP_POISON_B
        ).passed()) {
        fail("FP32 verifier accepted partial poisoned output");
    }

    poison_i32(device.output_i32, INT_POISON_A);
    gemv_integer_zero_extend_negative<<<
        grid_for_rows(negative_shape.outputs), BLOCK_THREADS
    >>>(
        device.output_i32.get(),
        reinterpret_cast<const uint16_t *>(
            device.weights_packed.get()
        ),
        device.activations_i32.get(),
        negative_shape.outputs,
        negative_shape.inputs,
        WEIGHT_FRACTION_BITS
    );
    check_launch("zero-extension negative control");
    CUDA_CHECK(cudaDeviceSynchronize());
    if (integer_output_matches(
            device.output_i32, fixture.expected_i32, host_i32
        )) {
        fail("zero-extension device negative control unexpectedly passed");
    }

    std::printf(
        "SELFTEST device_gates=PASS boundary_shapes=%zu "
        "literal_rows=9 floor_cases=15 skipped_negative=PASS "
        "partial_wide=PASS partial_packed=PASS partial_fp32=PASS "
        "zero_extend_negative=PASS poison_bits=PASS\n",
        std::size(BOUNDARY_SHAPES)
    );
}

void enqueue_lane(
    Lane lane,
    const Fixture &fixture,
    DeviceFixture &device
) {
    switch (lane) {
        case Lane::FP32:
            gemv_fp32_warp<<<
                grid_for_rows(fixture.shape.outputs), BLOCK_THREADS
            >>>(
                device.output_fp32.get(),
                device.weights_fp32.get(),
                device.activations_fp32.get(),
                fixture.shape.outputs,
                fixture.shape.inputs
            );
            break;
        case Lane::WIDE:
            gemv_integer_warp<<<
                grid_for_rows(fixture.shape.outputs), BLOCK_THREADS
            >>>(
                device.output_i32.get(),
                device.weights_wide.get(),
                device.activations_i32.get(),
                fixture.shape.outputs,
                fixture.shape.inputs,
                WEIGHT_FRACTION_BITS
            );
            break;
        case Lane::PACKED:
            gemv_integer_warp<<<
                grid_for_rows(fixture.shape.outputs), BLOCK_THREADS
            >>>(
                device.output_i32.get(),
                device.weights_packed.get(),
                device.activations_i32.get(),
                fixture.shape.outputs,
                fixture.shape.inputs,
                WEIGHT_FRACTION_BITS
            );
            break;
    }
}

void warm_lane(
    Lane lane,
    const Fixture &fixture,
    DeviceFixture &device
) {
    enqueue_lane(lane, fixture, device);
    check_launch(lane_name(lane));
}

float timed_lane(
    Lane lane,
    const Fixture &fixture,
    DeviceFixture &device,
    DeviceBuffer<uint64_t> &pressure,
    bool use_pressure,
    int lane_sample_index,
    std::vector<int32_t> &host_i32,
    std::vector<float> &host_fp32,
    FloatingVerdict *floating_observed,
    EventPair &events
) {
    const int32_t integer_poison =
        (lane_sample_index & 1) == 0
        ? INT_POISON_A
        : INT_POISON_B;
    const uint32_t floating_poison =
        (lane_sample_index & 1) == 0
        ? FP_POISON_A
        : FP_POISON_B;

    if (use_pressure) {
        apply_pressure(pressure);
    } else {
        warm_lane(lane, fixture, device);
    }

    if (lane == Lane::FP32) {
        poison_fp32(device.output_fp32, floating_poison);
    } else {
        poison_i32(device.output_i32, integer_poison);
    }

    CUDA_CHECK(cudaEventRecord(events.start()));
    enqueue_lane(lane, fixture, device);
    CUDA_CHECK(cudaEventRecord(events.stop()));
    check_launch(lane_name(lane));
    CUDA_CHECK(cudaEventSynchronize(events.stop()));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds, events.start(), events.stop()
    ));
    if (!std::isfinite(milliseconds) || milliseconds <= 0.0f) {
        fail("invalid CUDA event duration");
    }

    if (lane == Lane::FP32) {
        verify_fp32_output(
            device.output_fp32,
            fixture.expected_fp32,
            host_fp32,
            floating_poison,
            floating_observed
        );
    } else {
        verify_integer_output(
            device.output_i32,
            fixture.expected_i32,
            host_i32
        );
    }
    return milliseconds * 1000.0f;
}

double median(std::vector<float> values) {
    if (values.empty()) fail("median of empty vector");
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0) return values[middle];
    return (
        static_cast<double>(values[middle - 1])
        + static_cast<double>(values[middle])
    ) / 2.0;
}

struct RegimeSummary {
    double fp32_median_us = 0.0;
    double wide_median_us = 0.0;
    double packed_median_us = 0.0;
};

RegimeSummary run_regime(
    const char *regime,
    const Fixture &fixture,
    DeviceFixture &device,
    DeviceBuffer<uint64_t> &pressure,
    bool use_pressure,
    bool reverse_permutations,
    FloatingVerdict *floating_observed,
    EvidenceCounts *evidence
) {
    if (floating_observed == nullptr || evidence == nullptr) {
        fail("missing regime evidence state");
    }
    constexpr std::array<std::array<Lane, 3>, 6> PERMUTATIONS{{
        {{Lane::FP32, Lane::WIDE, Lane::PACKED}},
        {{Lane::FP32, Lane::PACKED, Lane::WIDE}},
        {{Lane::WIDE, Lane::FP32, Lane::PACKED}},
        {{Lane::WIDE, Lane::PACKED, Lane::FP32}},
        {{Lane::PACKED, Lane::FP32, Lane::WIDE}},
        {{Lane::PACKED, Lane::WIDE, Lane::FP32}},
    }};

    std::vector<float> fp32_samples;
    std::vector<float> wide_samples;
    std::vector<float> packed_samples;
    fp32_samples.reserve(12);
    wide_samples.reserve(12);
    packed_samples.reserve(12);
    std::vector<int32_t> host_i32(fixture.expected_i32.size());
    std::vector<float> host_fp32(fixture.expected_fp32.size());
    EventPair events;
    std::array<int, 3> lane_observations{{0, 0, 0}};

    for (int cycle = 0; cycle < 2; ++cycle) {
        for (int step = 0; step < 6; ++step) {
            const int permutation_index = reverse_permutations
                ? 5 - step
                : step;
            for (int position = 0; position < 3; ++position) {
                const Lane lane =
                    PERMUTATIONS[static_cast<std::size_t>(
                        permutation_index
                    )][static_cast<std::size_t>(position)];
                const std::size_t slot = lane_slot(lane);
                const int lane_sample_index =
                    lane_observations[slot];
                const float microseconds = timed_lane(
                    lane,
                    fixture,
                    device,
                    pressure,
                    use_pressure,
                    lane_sample_index,
                    host_i32,
                    host_fp32,
                    floating_observed,
                    events
                );
                ++lane_observations[slot];
                switch (lane) {
                    case Lane::FP32:
                        fp32_samples.push_back(microseconds);
                        break;
                    case Lane::WIDE:
                        wide_samples.push_back(microseconds);
                        break;
                    case Lane::PACKED:
                        packed_samples.push_back(microseconds);
                        break;
                }
                std::printf(
                    "SAMPLE shape=%s regime=%s cycle=%d "
                    "permutation=%d position=%d lane=%s "
                    "lane_observation=%d poison=%c "
                    "time_us=%.6f output_verified=true\n",
                    fixture.shape.id,
                    regime,
                    cycle,
                    permutation_index,
                    position,
                    lane_name(lane),
                    lane_sample_index,
                    (lane_sample_index & 1) == 0 ? 'A' : 'B',
                    static_cast<double>(microseconds)
                );
                ++evidence->samples;
            }
        }
    }
    if (fp32_samples.size() != 12
        || wide_samples.size() != 12
        || packed_samples.size() != 12
        || lane_observations[0] != 12
        || lane_observations[1] != 12
        || lane_observations[2] != 12) {
        fail("incomplete retained timing schedule");
    }

    RegimeSummary summary{
        median(fp32_samples),
        median(wide_samples),
        median(packed_samples),
    };
    std::printf(
        "SUMMARY shape=%s regime=%s samples_per_lane=12 "
        "fp32_median_us=%.6f wide_int32_median_us=%.6f "
        "packed_int16_median_us=%.6f "
        "wide_over_packed=%.6f fp32_over_packed=%.6f\n",
        fixture.shape.id,
        regime,
        summary.fp32_median_us,
        summary.wide_median_us,
        summary.packed_median_us,
        summary.wide_median_us / summary.packed_median_us,
        summary.fp32_median_us / summary.packed_median_us
    );
    ++evidence->summaries;
    return summary;
}

std::size_t pressure_bytes_for(const cudaDeviceProp &properties) {
    if (properties.l2CacheSize <= 0) {
        fail("target did not report a positive L2 size");
    }
    const std::size_t l2 =
        static_cast<std::size_t>(properties.l2CacheSize);
    if (l2 > std::numeric_limits<std::size_t>::max() / 2) {
        fail("2x L2 pressure size overflow");
    }
    const std::size_t required = std::max(
        MIN_PRESSURE_BYTES, l2 * 2
    );
    if (required
        > std::numeric_limits<std::size_t>::max()
            - (sizeof(uint64_t) - 1)) {
        fail("pressure alignment overflow");
    }
    return (
        required + sizeof(uint64_t) - 1
    ) / sizeof(uint64_t) * sizeof(uint64_t);
}

uint64_t verify_pressure_full_footprint(
    DeviceBuffer<uint64_t> &pressure
) {
    constexpr uint64_t INITIAL = UINT64_C(0x3535353535353535);
    constexpr uint64_t MULTIPLIER =
        UINT64_C(6364136223846793005);
    constexpr uint64_t INCREMENT =
        UINT64_C(1442695040888963407);
    constexpr uint64_t FNV_OFFSET =
        UINT64_C(1469598103934665603);
    constexpr uint64_t FNV_PRIME =
        UINT64_C(1099511628211);

    apply_pressure(pressure);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint64_t> observed(pressure.count());
    CUDA_CHECK(cudaMemcpy(
        observed.data(),
        pressure.get(),
        pressure.bytes(),
        cudaMemcpyDeviceToHost
    ));
    uint64_t checksum = FNV_OFFSET;
    for (std::size_t index = 0; index < observed.size(); ++index) {
        const uint64_t expected =
            INITIAL * MULTIPLIER
            + static_cast<uint64_t>(index)
            + INCREMENT;
        if (observed[index] != expected) {
            fail(
                std::string("pressure footprint mismatch at word ")
                + std::to_string(index)
            );
        }
        checksum ^= observed[index];
        checksum *= FNV_PRIME;
    }
    return checksum;
}

struct ShapeResult {
    std::string id;
    RegimeSummary no_pressure;
    RegimeSummary pressure;
};

ShapeResult run_shape(
    const Shape &shape,
    DeviceBuffer<uint64_t> &pressure,
    bool pressure_first,
    bool qualification_only,
    FloatingVerdict *floating_observed,
    EvidenceCounts *evidence
) {
    if (floating_observed == nullptr || evidence == nullptr) {
        fail("missing shape evidence state");
    }
    Fixture fixture = make_fixture(shape);
    DeviceFixture device(fixture);

    const double wide_h2d_us = copy_h2d_microseconds(
        device.weights_wide.get(),
        fixture.weights_wide.data(),
        device.weights_wide.bytes()
    );
    const double packed_h2d_us = copy_h2d_microseconds(
        device.weights_packed.get(),
        fixture.weights_packed.data(),
        device.weights_packed.bytes()
    );
    const double fp32_h2d_us = copy_h2d_microseconds(
        device.weights_fp32.get(),
        fixture.weights_fp32.data(),
        device.weights_fp32.bytes()
    );
    std::printf(
        "BOUNDARY shape=%s "
        "measurement=diagnostic_one_shot_pageable_host_wallclock "
        "scope=one_time_weight_materialization_and_transfer "
        "pack_us=%.3f "
        "wide_bytes=%zu packed_bytes=%zu fp32_bytes=%zu "
        "shared_activation_i32_bytes=%zu output_i32_bytes=%zu "
        "wide_h2d_us=%.3f packed_h2d_us=%.3f fp32_h2d_us=%.3f "
        "activation_conversion=none\n",
        shape.id,
        fixture.pack_microseconds,
        device.weights_wide.bytes(),
        device.weights_packed.bytes(),
        device.weights_fp32.bytes(),
        device.activations_i32.bytes(),
        device.output_i32.bytes(),
        wide_h2d_us,
        packed_h2d_us,
        fp32_h2d_us
    );
    ++evidence->boundaries;

    std::vector<int32_t> host_i32(fixture.expected_i32.size());
    std::vector<float> host_fp32(fixture.expected_fp32.size());
    require_integer_poison_distinct(
        fixture.expected_i32, INT_POISON_A
    );
    require_integer_poison_distinct(
        fixture.expected_i32, INT_POISON_B
    );
    FloatingVerdict shape_floating;

    poison_i32(device.output_i32, INT_POISON_A);
    launch_wide(
        device.output_i32.get(),
        device.weights_wide.get(),
        device.activations_i32.get(),
        shape.outputs,
        shape.inputs
    );
    verify_integer_output(
        device.output_i32, fixture.expected_i32, host_i32
    );
    poison_i32(device.output_i32, INT_POISON_B);
    launch_packed(
        device.output_i32.get(),
        device.weights_packed.get(),
        device.activations_i32.get(),
        shape.outputs,
        shape.inputs
    );
    verify_integer_output(
        device.output_i32, fixture.expected_i32, host_i32
    );
    poison_fp32(device.output_fp32, FP_POISON_A);
    launch_fp32(
        device.output_fp32.get(),
        device.weights_fp32.get(),
        device.activations_fp32.get(),
        shape.outputs,
        shape.inputs
    );
    verify_fp32_output(
        device.output_fp32,
        fixture.expected_fp32,
        host_fp32,
        FP_POISON_A,
        &shape_floating
    );
    std::printf(
        "PREFLIGHT shape=%s pack_identity=PASS "
        "signed128_oracle=PASS wide_output=PASS "
        "packed_output=PASS fp32_bound=PASS "
        "maximum_absolute_product_sum=%llu "
        "absolute_product_margin_to_i64_max=%llu "
        "maximum_absolute_exact_sum=%llu\n",
        shape.id,
        static_cast<unsigned long long>(
            fixture.maximum_absolute_product_sum
        ),
        static_cast<unsigned long long>(
            static_cast<uint64_t>(
                std::numeric_limits<int64_t>::max()
            ) - fixture.maximum_absolute_product_sum
        ),
        static_cast<unsigned long long>(
            fixture.maximum_absolute_exact_sum
        )
    );
    ++evidence->preflights;

    ShapeResult result{};
    result.id = shape.id;
    if (qualification_only) {
        if (!shape_floating.passed()
            || shape_floating.calls != 1
            || shape_floating.elements
                != static_cast<std::size_t>(shape.outputs)) {
            fail("qualification FP32 verification aggregate invalid");
        }
        floating_observed->merge(shape_floating);
        std::printf(
            "QUALIFICATION shape=%s wide_output=PASS "
            "packed_output=PASS fp32_bound=PASS\n",
            shape.id
        );
        ++evidence->shapes;
        return result;
    }
    auto no_pressure = [&]() {
        result.no_pressure = run_regime(
            "repeat_hot",
            fixture,
            device,
            pressure,
            false,
            false,
            &shape_floating,
            evidence
        );
    };
    auto after_pressure = [&]() {
        result.pressure = run_regime(
            "after_l2_pressure",
            fixture,
            device,
            pressure,
            true,
            true,
            &shape_floating,
            evidence
        );
    };
    if (pressure_first) {
        after_pressure();
        no_pressure();
    } else {
        no_pressure();
        after_pressure();
    }

    const std::size_t expected_fp32_calls = 25;
    const std::size_t expected_fp32_elements =
        static_cast<std::size_t>(shape.outputs)
        * expected_fp32_calls;
    if (!shape_floating.passed()
        || shape_floating.calls != expected_fp32_calls
        || shape_floating.elements != expected_fp32_elements) {
        fail("shape-local FP32 verification aggregate invalid");
    }
    floating_observed->merge(shape_floating);
    std::printf(
        "POST shape=%s timed_outputs=PASS "
        "fp32_calls=%zu fp32_elements=%zu "
        "fp32_max_error=%.12Le fp32_max_bound=%.12Le\n",
        shape.id,
        shape_floating.calls,
        shape_floating.elements,
        shape_floating.maximum_error,
        shape_floating.maximum_bound
    );
    ++evidence->posts;
    ++evidence->shapes;
    return result;
}

std::string hex_bytes(const unsigned char *data, std::size_t count) {
    if (data == nullptr || count == 0) fail("invalid hex input");
    static constexpr char DIGITS[] = "0123456789abcdef";
    std::string encoded;
    encoded.reserve(count * 2);
    for (std::size_t index = 0; index < count; ++index) {
        encoded.push_back(DIGITS[data[index] >> 4]);
        encoded.push_back(DIGITS[data[index] & UINT8_C(0x0f)]);
    }
    return encoded;
}

void require_locked_build_identity() {
    if (std::strcmp(P13_BENCH_SOURCE_SHA256, "UNLOCKED") == 0
        || std::strcmp(P13_MATH_HEADER_SHA256, "UNLOCKED") == 0
        || std::strcmp(P13_SELFTEST_HEADER_SHA256, "UNLOCKED") == 0
        || std::strcmp(P13_CANDIDATE_SHA256, "UNLOCKED") == 0
        || std::strcmp(P13_BUILD_ID, "UNLOCKED") == 0) {
        fail("benchmark was not built by the hash-locked build");
    }
}

struct RunIdentity {
    int replicate_id = -1;
    std::string run_id;
};

RunIdentity require_run_identity() {
    const char *replicate_text = std::getenv("P13_REPLICATE_ID");
    const char *run_text = std::getenv("P13_RUN_ID");
    if (replicate_text == nullptr
        || replicate_text[0] < '0'
        || replicate_text[0] > '2'
        || replicate_text[1] != '\0') {
        fail("P13_REPLICATE_ID must be exactly 0, 1 or 2");
    }
    if (run_text == nullptr || std::strlen(run_text) != 32) {
        fail("P13_RUN_ID must be 32 lowercase hexadecimal characters");
    }
    for (std::size_t index = 0; index < 32; ++index) {
        const char character = run_text[index];
        if (!((character >= '0' && character <= '9')
            || (character >= 'a' && character <= 'f'))) {
            fail("P13_RUN_ID must be lowercase hexadecimal");
        }
    }
    return {
        replicate_text[0] - '0',
        std::string(run_text),
    };
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const bool qualification_only =
            argc == 2
            && std::strcmp(argv[1], "--qualify-only") == 0;
        if (argc != 1 && !qualification_only) {
            fail("usage: bench_p13_int16 [--qualify-only]");
        }
        require_locked_build_identity();
        const RunIdentity run_identity = require_run_identity();
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        if (properties.major != 12 || properties.minor != 0) {
            fail("target must be compute capability 12.0");
        }
        if (properties.warpSize != static_cast<int>(WARP_SIZE)) {
            fail("target warp size is not 32");
        }
        int runtime_version = 0;
        int driver_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        const std::string gpu_name_hex = hex_bytes(
            reinterpret_cast<const unsigned char *>(properties.name),
            std::strlen(properties.name)
        );
        const std::string device_uuid = hex_bytes(
            reinterpret_cast<const unsigned char *>(
                properties.uuid.bytes
            ),
            sizeof(properties.uuid.bytes)
        );
        std::printf(
            "PROVENANCE schema=p13_int16_v2 gpu_name_hex=%s "
            "device_uuid=%s sm_count=%d compute=%d.%d "
            "l2_bytes=%d warp_size=%d nvcc_cudart=%d "
            "runtime_version=%d driver_version=%d "
            "source_sha256=%s math_header_sha256=%s "
            "selftest_header_sha256=%s candidate_sha256=%s "
            "build_id=%s replicate_id=%d run_id=%s\n",
            gpu_name_hex.c_str(),
            device_uuid.c_str(),
            properties.multiProcessorCount,
            properties.major,
            properties.minor,
            properties.l2CacheSize,
            properties.warpSize,
            CUDART_VERSION,
            runtime_version,
            driver_version,
            P13_BENCH_SOURCE_SHA256,
            P13_MATH_HEADER_SHA256,
            P13_SELFTEST_HEADER_SHA256,
            P13_CANDIDATE_SHA256,
            P13_BUILD_ID,
            run_identity.replicate_id,
            run_identity.run_id.c_str()
        );
        std::printf(
            "SEMANTICS weight_storage=wide_i32_vs_packed_i16 "
            "weight_fraction_bits=%u activation=i32_f16 "
            "accumulator=signed_i64 oracle=signed_i128 "
            "epilogue=mathematical_floor no_tensor_cores=true\n",
            WEIGHT_FRACTION_BITS
        );

        run_host_controls();
        run_device_selftests();

        const std::size_t pressure_bytes =
            pressure_bytes_for(properties);
        const std::size_t l2_bytes =
            static_cast<std::size_t>(properties.l2CacheSize);
        if (pressure_bytes < l2_bytes * 2) {
            fail("pressure footprint is below 2x reported L2");
        }
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        DeviceBuffer<uint64_t> pressure(
            pressure_bytes / sizeof(uint64_t)
        );
        CUDA_CHECK(cudaMemset(
            pressure.get(), 0x35, pressure.bytes()
        ));
        const uint64_t pressure_checksum =
            verify_pressure_full_footprint(pressure);
        std::printf(
            "PRESSURE bytes=%zu l2_bytes=%zu "
            "ratio_to_l2=%.6f verified_words=%zu "
            "full_footprint=PASS checksum_fnv64=%016llx "
            "free_before_bytes=%zu total_bytes=%zu\n",
            pressure.bytes(),
            l2_bytes,
            static_cast<double>(pressure.bytes())
                / static_cast<double>(l2_bytes),
            pressure.count(),
            static_cast<unsigned long long>(pressure_checksum),
            free_bytes,
            total_bytes
        );

        std::vector<ShapeResult> results;
        FloatingVerdict floating_observed;
        EvidenceCounts evidence;
        for (std::size_t index = 0;
             index < std::size(SHAPES);
             ++index) {
            results.push_back(run_shape(
                SHAPES[index],
                pressure,
                (index & 1U) != 0,
                qualification_only,
                &floating_observed,
                &evidence
            ));
        }
        if (qualification_only) {
            std::size_t expected_fp32_elements = 0;
            for (const Shape &shape : SHAPES) {
                expected_fp32_elements +=
                    static_cast<std::size_t>(shape.outputs);
            }
            if (!floating_observed.passed()
                || floating_observed.calls != std::size(SHAPES)
                || floating_observed.elements
                    != expected_fp32_elements
                || results.size() != std::size(SHAPES)
                || evidence.samples != 0
                || evidence.summaries != 0
                || evidence.boundaries != 4
                || evidence.preflights != 4
                || evidence.posts != 0
                || evidence.shapes != 4) {
                fail("incomplete qualification evidence");
            }
            std::printf(
                "FINAL disposition=QUALIFICATION_PASS "
                "scope=non_timed_sanitizer_gate shapes=%zu "
                "boundaries=%zu preflights=%zu fp32_calls=%zu "
                "fp32_elements=%zu retained_timing=none\n",
                evidence.shapes,
                evidence.boundaries,
                evidence.preflights,
                floating_observed.calls,
                floating_observed.elements
            );
            if (std::fflush(stdout) != 0
                || std::ferror(stdout) != 0) {
                fail("stdout qualification flush failed");
            }
            return 0;
        }
        const std::size_t expected_fp32_calls =
            std::size(SHAPES) * 25;
        std::size_t expected_fp32_elements = 0;
        for (const Shape &shape : SHAPES) {
            expected_fp32_elements +=
                static_cast<std::size_t>(shape.outputs) * 25;
        }
        if (!floating_observed.passed()
            || floating_observed.calls != expected_fp32_calls
            || floating_observed.elements != expected_fp32_elements) {
            fail("aggregate FP32 verification state invalid");
        }
        if (results.size() != std::size(SHAPES)
            || evidence.samples != 288
            || evidence.summaries != 8
            || evidence.boundaries != 4
            || evidence.preflights != 4
            || evidence.posts != 4
            || evidence.shapes != 4) {
            fail("incomplete evidence record counts");
        }
        for (std::size_t index = 0; index < results.size(); ++index) {
            if (results[index].id != SHAPES[index].id) {
                fail("shape result identity/order mismatch");
            }
        }
        std::printf(
            "DIMENSIONS representation_identity=PASS "
            "integer_exactness=PASS floating_bound=PASS "
            "timed_output_integrity=PASS "
            "boundary_disclosure=DIAGNOSTIC_ONLY "
            "automatic_promotion=DISABLED\n"
        );
        std::printf(
            "FINAL disposition=MEASURED_NEEDS_EXTERNAL_VALIDATION "
            "scope=synthetic_m1_projection_gemv "
            "classification=representation_plus_execution "
            "samples=%zu summaries=%zu boundaries=%zu "
            "preflights=%zu posts=%zu shapes=%zu "
            "fp32_calls=%zu fp32_elements=%zu "
            "nonclaims=no_layer_no_tokens_no_runtime_no_tensor_cores\n",
            evidence.samples,
            evidence.summaries,
            evidence.boundaries,
            evidence.preflights,
            evidence.posts,
            evidence.shapes,
            floating_observed.calls,
            floating_observed.elements
        );
        if (std::fflush(stdout) != 0 || std::ferror(stdout) != 0) {
            fail("stdout evidence flush failed");
        }
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
