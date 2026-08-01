#ifndef INT_LLM_REAL_WEIGHT_MATH_H
#define INT_LLM_REAL_WEIGHT_MATH_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace real_weight {

using i128 = __int128;
using u128 = unsigned __int128;

constexpr int FRACTION_BITS = 48;
constexpr i128 Q48_SCALE = static_cast<i128>(1) << FRACTION_BITS;
constexpr u128 I128_MAX_U = (static_cast<u128>(1) << 127) - 1;

struct DotReference {
    int64_t q48;
    long double exact_real;
    long double absolute_real_sum;
};

struct FloatingReference {
    long double exact_real;
    long double absolute_real_sum;
};

struct Bf16Component {
    int16_t significand;
    int16_t exponent;
};

struct IntegerVerdict {
    std::size_t elements;
    std::size_t mismatches;
    std::size_t first_mismatch;

    bool passed() const { return elements != 0 && mismatches == 0; }
};

struct FloatingVerdict {
    std::size_t elements;
    std::size_t violations;
    std::size_t sentinel_hits;
    std::size_t first_violation;
    long double maximum_error;
    long double maximum_bound;

    bool passed() const { return elements != 0 && violations == 0; }
};

inline bool bf16_to_q48(uint16_t bits, int64_t *out) {
    if (out == nullptr) return false;
    const bool negative = (bits & UINT16_C(0x8000)) != 0;
    const unsigned exponent = (bits >> 7) & 0xffu;
    const unsigned mantissa = bits & 0x7fu;

    if (exponent == 0xffu) return false;
    if (exponent == 0u) {
        *out = 0;
        return true;
    }

    const unsigned significand = 128u + mantissa;
    const int shift = static_cast<int>(exponent) - 86;
    const u128 limit = negative
        ? (static_cast<u128>(1) << 63)
        : static_cast<u128>(std::numeric_limits<int64_t>::max());
    u128 magnitude = 0;
    if (shift >= 0) {
        if (shift >= 128
            || static_cast<u128>(significand) > (limit >> shift)) {
            return false;
        }
        magnitude = static_cast<u128>(significand) << shift;
    } else {
        const int discarded_bits = -shift;
        magnitude = discarded_bits >= 8
            ? 0
            : static_cast<u128>(significand) >> discarded_bits;
    }

    if (negative && magnitude == (static_cast<u128>(1) << 63)) {
        *out = std::numeric_limits<int64_t>::min();
    } else {
        const int64_t signed_magnitude = static_cast<int64_t>(magnitude);
        *out = negative ? -signed_magnitude : signed_magnitude;
    }
    return true;
}

inline bool bf16_is_exact_q48(uint16_t bits) {
    int64_t ignored = 0;
    if (!bf16_to_q48(bits, &ignored)) return false;
    const unsigned exponent = (bits >> 7) & 0xffu;
    const unsigned mantissa = bits & 0x7fu;
    if (exponent == 0u) return mantissa == 0u;
    const int shift = static_cast<int>(exponent) - 86;
    if (shift >= 0) return true;
    const int discarded_bits = -shift;
    if (discarded_bits >= 8) return false;
    const unsigned mask = (1u << discarded_bits) - 1u;
    return ((128u + mantissa) & mask) == 0u;
}

/*
 * Independent host oracle for the conversion gate. This deliberately uses
 * power-of-two scaling and truncation in long double instead of the production
 * shift implementation above. On the finite, Q16.48-range domain accepted by
 * the safe wrapper, it reproduces the repository's BF16-to-Q16.48 rule,
 * including truncation toward zero below the Q48 grid.
 */
inline bool bf16_to_q48_independent(uint16_t bits, int64_t *out) {
    if (out == nullptr) return false;
    const bool negative = (bits & UINT16_C(0x8000)) != 0;
    const unsigned exponent = (bits >> 7) & 0xffu;
    const unsigned mantissa = bits & 0x7fu;
    if (exponent == 0xffu) return false;
    const unsigned significand =
        exponent == 0u ? mantissa : 128u + mantissa;
    const int scaled_exponent =
        exponent == 0u ? -85 : static_cast<int>(exponent) - 86;
    long double scaled = std::ldexp(
        static_cast<long double>(significand), scaled_exponent
    );
    if (negative) scaled = -scaled;
    const long double positive_exclusive = 0x1p63L;
    const long double negative_inclusive = -0x1p63L;
    if (!std::isfinite(scaled)
        || scaled >= positive_exclusive
        || scaled < negative_inclusive) {
        return false;
    }
    const long double truncated = std::trunc(scaled);
    if (truncated == negative_inclusive) {
        *out = std::numeric_limits<int64_t>::min();
    } else {
        *out = static_cast<int64_t>(truncated);
    }
    return true;
}

inline bool bf16_component(uint16_t bits, Bf16Component *out) {
    if (out == nullptr) return false;
    const bool negative = (bits & UINT16_C(0x8000)) != 0;
    const unsigned exponent = (bits >> 7) & 0xffu;
    const unsigned mantissa = bits & 0x7fu;
    if (exponent == 0xffu) return false;
    if (exponent == 0u && mantissa == 0u) {
        *out = {0, 0};
        return true;
    }
    const int significand = exponent == 0u
        ? static_cast<int>(mantissa)
        : static_cast<int>(128u + mantissa);
    out->significand = static_cast<int16_t>(
        negative ? -significand : significand
    );
    out->exponent = static_cast<int16_t>(
        exponent == 0u ? -133 : static_cast<int>(exponent) - 134
    );
    return true;
}

inline u128 magnitude(i128 value) {
    if (value >= 0) return static_cast<u128>(value);
    return static_cast<u128>(-(value + 1)) + 1;
}

inline bool floor_shift_q48(i128 value, int64_t *out) {
    if (out == nullptr) return false;
    i128 shifted;
    if (value >= 0) {
        shifted = value / Q48_SCALE;
    } else {
        const u128 rounded =
            (magnitude(value) + static_cast<u128>(Q48_SCALE - 1))
            / static_cast<u128>(Q48_SCALE);
        if (rounded > (static_cast<u128>(1) << 63)) return false;
        if (rounded == (static_cast<u128>(1) << 63)) {
            *out = std::numeric_limits<int64_t>::min();
            return true;
        }
        shifted = -static_cast<i128>(rounded);
    }
    if (shifted < static_cast<i128>(std::numeric_limits<int64_t>::min())
        || shifted
            > static_cast<i128>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    *out = static_cast<int64_t>(shifted);
    return true;
}

inline bool dot_reference(
    const int64_t *input,
    const int64_t *weights,
    std::size_t count,
    DotReference *out
) {
    if (input == nullptr || weights == nullptr || out == nullptr) return false;
    i128 sum = 0;
    u128 absolute_sum = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const i128 product =
            static_cast<i128>(input[index])
            * static_cast<i128>(weights[index]);
        const u128 product_magnitude = magnitude(product);
        if (product_magnitude > I128_MAX_U - absolute_sum) return false;
        absolute_sum += product_magnitude;
        sum += product;
    }
    if (!floor_shift_q48(sum, &out->q48)) return false;

    const long double product_scale =
        static_cast<long double>(Q48_SCALE)
        * static_cast<long double>(Q48_SCALE);
    out->exact_real = static_cast<long double>(sum) / product_scale;
    out->absolute_real_sum =
        static_cast<long double>(absolute_sum) / product_scale;
    return true;
}

inline bool bf16_component_dot_reference(
    const Bf16Component *input,
    const Bf16Component *weights,
    std::size_t count,
    FloatingReference *out
) {
    if (input == nullptr || weights == nullptr || out == nullptr || count == 0) {
        return false;
    }
    int minimum_exponent = std::numeric_limits<int>::max();
    for (std::size_t index = 0; index < count; ++index) {
        if (input[index].significand != 0
            && weights[index].significand != 0) {
            minimum_exponent = std::min(
                minimum_exponent,
                static_cast<int>(input[index].exponent)
                    + static_cast<int>(weights[index].exponent)
            );
        }
    }
    if (minimum_exponent == std::numeric_limits<int>::max()) {
        *out = {0.0L, 0.0L};
        return true;
    }

    i128 exact_sum = 0;
    u128 absolute_sum = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const int product_significand =
            static_cast<int>(input[index].significand)
            * static_cast<int>(weights[index].significand);
        if (product_significand == 0) continue;
        const int product_exponent =
            static_cast<int>(input[index].exponent)
            + static_cast<int>(weights[index].exponent);
        const int shift = product_exponent - minimum_exponent;
        const uint32_t small_magnitude = static_cast<uint32_t>(
            product_significand < 0
                ? -product_significand
                : product_significand
        );
        if (shift < 0 || shift >= 128
            || static_cast<u128>(small_magnitude)
                > (I128_MAX_U >> shift)) {
            return false;
        }
        const u128 term = static_cast<u128>(small_magnitude) << shift;
        if (term > I128_MAX_U - absolute_sum) return false;
        absolute_sum += term;
        if (product_significand < 0) {
            exact_sum -= static_cast<i128>(term);
        } else {
            exact_sum += static_cast<i128>(term);
        }
    }
    out->exact_real = std::ldexp(
        static_cast<long double>(exact_sum), minimum_exponent
    );
    out->absolute_real_sum = std::ldexp(
        static_cast<long double>(absolute_sum), minimum_exponent
    );
    return true;
}

inline long double fp32_accumulation_bound(
    std::size_t products,
    long double absolute_real_sum
) {
    if (products == 0) return 0.0L;
    const long double unit_roundoff = 0x1p-24L;
    const long double stages =
        2.0L * static_cast<long double>(products);
    const long double denominator = 1.0L - stages * unit_roundoff;
    if (denominator <= 0.0L) {
        return std::numeric_limits<long double>::infinity();
    }
    const long double gamma = stages * unit_roundoff / denominator;
    return 8.0L * gamma * absolute_real_sum + 0x1p-20L;
}

inline IntegerVerdict verify_integer_values(
    const int64_t *actual,
    const DotReference *expected,
    std::size_t count
) {
    IntegerVerdict verdict{count, 0, 0};
    if (actual == nullptr || expected == nullptr || count == 0) {
        verdict.mismatches = count == 0 ? 1 : count;
        return verdict;
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (actual[index] != expected[index].q48) {
            if (verdict.mismatches == 0) verdict.first_mismatch = index;
            ++verdict.mismatches;
        }
    }
    return verdict;
}

inline FloatingVerdict verify_floating_values(
    const float *actual,
    const FloatingReference *expected,
    std::size_t count,
    std::size_t products,
    uint32_t forbidden_bits = 0,
    bool enforce_forbidden_bits = false
) {
    FloatingVerdict verdict{count, 0, 0, 0, 0.0L, 0.0L};
    if (actual == nullptr || expected == nullptr
        || count == 0 || products == 0) {
        verdict.violations = count == 0 ? 1 : count;
        return verdict;
    }
    for (std::size_t index = 0; index < count; ++index) {
        const long double observed = static_cast<long double>(actual[index]);
        const long double error =
            std::fabs(observed - expected[index].exact_real);
        const long double bound = fp32_accumulation_bound(
            products, expected[index].absolute_real_sum
        );
        uint32_t actual_bits = 0;
        static_assert(sizeof(actual_bits) == sizeof(actual[index]),
                      "float32 representation");
        std::memcpy(&actual_bits, &actual[index], sizeof(actual_bits));
        const bool sentinel_hit =
            enforce_forbidden_bits && actual_bits == forbidden_bits;
        if (sentinel_hit) ++verdict.sentinel_hits;
        if (sentinel_hit || !std::isfinite(actual[index]) || error > bound) {
            if (verdict.violations == 0) verdict.first_violation = index;
            ++verdict.violations;
        }
        verdict.maximum_error = std::max(verdict.maximum_error, error);
        verdict.maximum_bound = std::max(verdict.maximum_bound, bound);
    }
    return verdict;
}

inline bool target_environment_matches(
    const char *gpu_name,
    int compute_major,
    int compute_minor,
    std::size_t total_global_memory,
    int cuda_runtime_version
) {
    if (gpu_name == nullptr) return false;
    const int runtime_major = cuda_runtime_version / 1000;
    const int runtime_minor = (cuda_runtime_version % 1000) / 10;
    return std::strcmp(gpu_name, "NVIDIA GeForce RTX 5090") == 0
        && compute_major == 12
        && compute_minor == 0
        && total_global_memory >= (30ULL << 30)
        && runtime_major == 13
        && runtime_minor == 3;
}

}  // namespace real_weight

#endif
