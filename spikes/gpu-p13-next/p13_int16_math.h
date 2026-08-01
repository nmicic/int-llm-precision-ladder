#ifndef INT_LLM_P13_INT16_MATH_H
#define INT_LLM_P13_INT16_MATH_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace p13_int16 {

using i128 = __int128;
using u128 = unsigned __int128;

constexpr u128 I128_MAX_U = (static_cast<u128>(1) << 127) - 1;

struct DotReference {
    int64_t exact_sum;
    uint64_t absolute_product_sum;
    int32_t output;
};

struct VerifyVerdict {
    std::size_t elements;
    std::size_t mismatches;
    std::size_t first_mismatch;

    bool passed() const {
        return elements != 0 && mismatches == 0;
    }
};

inline u128 magnitude(i128 value) {
    if (value >= 0) return static_cast<u128>(value);
    return static_cast<u128>(-(value + 1)) + 1;
}

inline bool fits_signed_width(int32_t value, unsigned bits) {
    if (bits == 0 || bits > 15) return false;
    const int32_t limit = static_cast<int32_t>(UINT32_C(1) << (bits - 1));
    return value >= -limit && value <= limit - 1;
}

inline bool pack_weight(
    int32_t canonical,
    unsigned declared_bits,
    int16_t *packed
) {
    if (packed == nullptr
        || !fits_signed_width(canonical, declared_bits)) {
        return false;
    }
    *packed = static_cast<int16_t>(canonical);
    return static_cast<int32_t>(*packed) == canonical;
}

inline int32_t decode_weight(int16_t packed) {
    return static_cast<int32_t>(packed);
}

inline bool floor_shift_to_i32(
    i128 value,
    unsigned fraction_bits,
    int32_t *output
) {
    if (output == nullptr || fraction_bits >= 63) return false;
    const u128 scale = static_cast<u128>(1) << fraction_bits;
    if (value >= 0) {
        const u128 quotient = static_cast<u128>(value) / scale;
        if (quotient
            > static_cast<u128>(
                std::numeric_limits<int32_t>::max()
            )) {
            return false;
        }
        *output = static_cast<int32_t>(quotient);
        return true;
    }

    const u128 value_magnitude = magnitude(value);
    u128 quotient = value_magnitude / scale;
    if ((value_magnitude % scale) != 0) ++quotient;
    const u128 negative_limit = static_cast<u128>(1) << 31;
    if (quotient > negative_limit) {
        return false;
    }
    if (quotient == negative_limit) {
        *output = std::numeric_limits<int32_t>::min();
        return true;
    }
    *output = -static_cast<int32_t>(quotient);
    return true;
}

inline bool dot_reference(
    const int32_t *canonical_weights,
    const int32_t *activations,
    std::size_t count,
    unsigned weight_fraction_bits,
    DotReference *output
) {
    if (canonical_weights == nullptr
        || activations == nullptr
        || output == nullptr
        || count == 0) {
        return false;
    }
    i128 sum = 0;
    u128 absolute_sum = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const i128 product =
            static_cast<i128>(canonical_weights[index])
            * static_cast<i128>(activations[index]);
        const u128 product_magnitude = magnitude(product);
        if (product_magnitude > I128_MAX_U - absolute_sum) return false;
        absolute_sum += product_magnitude;
        sum += product;
    }
    if (sum
            < static_cast<i128>(std::numeric_limits<int64_t>::min())
        || sum
            > static_cast<i128>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    if (absolute_sum
        > static_cast<u128>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    int32_t shifted = 0;
    if (!floor_shift_to_i32(sum, weight_fraction_bits, &shifted)) {
        return false;
    }
    output->exact_sum = static_cast<int64_t>(sum);
    output->absolute_product_sum =
        static_cast<uint64_t>(absolute_sum);
    output->output = shifted;
    return true;
}

inline bool dot_reference_packed(
    const int16_t *packed_weights,
    const int32_t *activations,
    std::size_t count,
    unsigned weight_fraction_bits,
    DotReference *output
) {
    if (packed_weights == nullptr
        || activations == nullptr
        || output == nullptr
        || count == 0) {
        return false;
    }
    i128 sum = 0;
    u128 absolute_sum = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const i128 product =
            static_cast<i128>(decode_weight(packed_weights[index]))
            * static_cast<i128>(activations[index]);
        const u128 product_magnitude = magnitude(product);
        if (product_magnitude > I128_MAX_U - absolute_sum) return false;
        absolute_sum += product_magnitude;
        sum += product;
    }
    if (sum
            < static_cast<i128>(std::numeric_limits<int64_t>::min())
        || sum
            > static_cast<i128>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    if (absolute_sum
        > static_cast<u128>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    int32_t shifted = 0;
    if (!floor_shift_to_i32(sum, weight_fraction_bits, &shifted)) {
        return false;
    }
    output->exact_sum = static_cast<int64_t>(sum);
    output->absolute_product_sum =
        static_cast<uint64_t>(absolute_sum);
    output->output = shifted;
    return true;
}

inline VerifyVerdict verify_i32(
    const int32_t *actual,
    const int32_t *expected,
    std::size_t count
) {
    VerifyVerdict verdict{count, 0, count};
    if (actual == nullptr || expected == nullptr || count == 0) {
        verdict.elements = 0;
        verdict.mismatches = 1;
        verdict.first_mismatch = 0;
        return verdict;
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (actual[index] != expected[index]) {
            if (verdict.mismatches == 0) {
                verdict.first_mismatch = index;
            }
            ++verdict.mismatches;
        }
    }
    return verdict;
}

}  // namespace p13_int16

#endif
