#ifndef INT_LLM_P13_INT16_SELFTEST_H
#define INT_LLM_P13_INT16_SELFTEST_H

#include "p13_int16_math.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace p13_int16 {

struct HostSelfTestResult {
    std::size_t checks = 0;
};

class HostSelfTest {
public:
    void require(bool condition, const char *message) {
        ++result_.checks;
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    HostSelfTestResult result() const {
        return result_;
    }

private:
    HostSelfTestResult result_{};
};

inline HostSelfTestResult run_host_math_selftests() {
    HostSelfTest test;

    for (unsigned bits = 9; bits <= 15; ++bits) {
        const int32_t limit =
            static_cast<int32_t>(UINT32_C(1) << (bits - 1));
        test.require(
            fits_signed_width(-limit, bits),
            "minimum does not fit declared signed width"
        );
        test.require(
            fits_signed_width(limit - 1, bits),
            "maximum does not fit declared signed width"
        );
        test.require(
            !fits_signed_width(-limit - 1, bits),
            "value below declared signed width was accepted"
        );
        test.require(
            !fits_signed_width(limit, bits),
            "value above declared signed width was accepted"
        );

        int16_t packed = 0;
        test.require(
            pack_weight(-limit, bits, &packed)
                && decode_weight(packed) == -limit,
            "minimum pack/decode identity failed"
        );
        test.require(
            pack_weight(limit - 1, bits, &packed)
                && decode_weight(packed) == limit - 1,
            "maximum pack/decode identity failed"
        );
    }

    test.require(
        !fits_signed_width(0, 0)
            && !fits_signed_width(0, 16)
            && !fits_signed_width(0, 17),
        "candidate-external signed widths were accepted"
    );
    int16_t packed_sentinel = INT16_C(1234);
    test.require(
        !pack_weight(-257, 9, &packed_sentinel)
            && packed_sentinel == INT16_C(1234),
        "9-bit underflow did not reject without mutation"
    );
    test.require(
        !pack_weight(256, 9, &packed_sentinel)
            && packed_sentinel == INT16_C(1234),
        "9-bit overflow did not reject without mutation"
    );
    test.require(
        !pack_weight(0, 9, nullptr),
        "null pack destination was accepted"
    );

    int16_t packed_negative = 0;
    test.require(
        pack_weight(-256, 9, &packed_negative)
            && decode_weight(packed_negative) == -256,
        "9-bit negative fixture did not pack losslessly"
    );
    test.require(
        !fits_signed_width(-256, 8),
        "the 9-bit negative fixture unexpectedly fits signed int8"
    );
    const uint8_t serialized_low_byte = static_cast<uint8_t>(
        static_cast<uint16_t>(packed_negative) & UINT16_C(0x00ff)
    );
    const int32_t explicit_int8_decode =
        (serialized_low_byte & UINT8_C(0x80)) != 0
        ? static_cast<int32_t>(serialized_low_byte) - 256
        : static_cast<int32_t>(serialized_low_byte);
    test.require(
        explicit_int8_decode != -256,
        "explicit signed-int8 truncation negative control passed"
    );

    struct FloorCase {
        i128 value;
        int32_t expected;
    };
    const std::array<FloorCase, 9> floor_cases{{
        {0, 0},
        {1, 0},
        {4095, 0},
        {4096, 1},
        {4097, 1},
        {-1, -1},
        {-4095, -1},
        {-4096, -1},
        {-4097, -2},
    }};
    for (const FloorCase &floor_case : floor_cases) {
        int32_t output = INT32_C(0x12345678);
        test.require(
            floor_shift_to_i32(floor_case.value, 12, &output)
                && output == floor_case.expected,
            "mathematical floor fixture failed"
        );
    }

    const i128 scale = static_cast<i128>(1) << 12;
    const i128 positive_max =
        static_cast<i128>(std::numeric_limits<int32_t>::max())
        * scale + scale - 1;
    const i128 positive_over = positive_max + 1;
    const i128 negative_min =
        static_cast<i128>(std::numeric_limits<int32_t>::min())
        * scale;
    const i128 negative_under = negative_min - 1;
    int32_t floor_output = INT32_C(0x12345678);
    test.require(
        floor_shift_to_i32(positive_max, 12, &floor_output)
            && floor_output == std::numeric_limits<int32_t>::max(),
        "positive int32 floor endpoint failed"
    );
    floor_output = INT32_C(0x12345678);
    test.require(
        !floor_shift_to_i32(positive_over, 12, &floor_output)
            && floor_output == INT32_C(0x12345678),
        "positive int32 floor overflow did not reject without mutation"
    );
    floor_output = INT32_C(0x12345678);
    test.require(
        floor_shift_to_i32(negative_min, 12, &floor_output)
            && floor_output == std::numeric_limits<int32_t>::min(),
        "negative int32 floor endpoint failed"
    );
    floor_output = INT32_C(0x12345678);
    test.require(
        !floor_shift_to_i32(negative_under, 12, &floor_output)
            && floor_output == INT32_C(0x12345678),
        "negative int32 floor overflow did not reject without mutation"
    );

    const i128 i128_min =
        -static_cast<i128>(I128_MAX_U) - static_cast<i128>(1);
    for (const unsigned bits : {0U, 12U, 62U}) {
        floor_output = INT32_C(0x12345678);
        test.require(
            !floor_shift_to_i32(i128_min, bits, &floor_output)
                && floor_output == INT32_C(0x12345678),
            "i128 minimum did not reject safely"
        );
    }
    floor_output = INT32_C(0x12345678);
    test.require(
        !floor_shift_to_i32(0, 63, &floor_output)
            && floor_output == INT32_C(0x12345678),
        "invalid floor shift count did not reject without mutation"
    );
    test.require(
        !floor_shift_to_i32(0, 12, nullptr),
        "null floor output was accepted"
    );

    const int32_t cancellation_weights[] = {
        16383, -16384, 1, -1
    };
    const int32_t cancellation_activations[] = {
        131072, 131071, 200000, -200001
    };
    int16_t cancellation_packed[4]{};
    for (std::size_t index = 0; index < 4; ++index) {
        test.require(
            pack_weight(
                cancellation_weights[index],
                15,
                &cancellation_packed[index]
            ),
            "cancellation fixture did not pack"
        );
    }
    DotReference wide{};
    DotReference narrow{};
    test.require(
        dot_reference(
            cancellation_weights,
            cancellation_activations,
            4,
            12,
            &wide
        ),
        "wide cancellation oracle rejected fixture"
    );
    test.require(
        dot_reference_packed(
            cancellation_packed,
            cancellation_activations,
            4,
            12,
            &narrow
        ),
        "packed cancellation oracle rejected fixture"
    );
    test.require(
        wide.exact_sum == INT64_C(285313)
            && wide.absolute_product_sum == UINT64_C(4295219841)
            && wide.output == 69,
        "hard-coded cancellation answer failed"
    );
    test.require(
        narrow.exact_sum == wide.exact_sum
            && narrow.absolute_product_sum
                == wide.absolute_product_sum
            && narrow.output == wide.output,
        "wide and packed cancellation oracles disagree"
    );
    test.require(
        wide.absolute_product_sum
            > static_cast<uint64_t>(wide.exact_sum),
        "cancellation fixture did not contain opposing products"
    );
    test.require(
        cancellation_activations[0]
            > std::numeric_limits<int16_t>::max()
            && cancellation_activations[3]
                < std::numeric_limits<int16_t>::min(),
        "cancellation fixture did not preserve int32-only activations"
    );

    const int32_t excessive_weights[] = {
        std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::max(),
        -std::numeric_limits<int32_t>::max(),
        -std::numeric_limits<int32_t>::max(),
    };
    const int32_t excessive_activations[] = {
        std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::max(),
    };
    DotReference rejected{
        INT64_C(123), UINT64_C(456), INT32_C(789)
    };
    test.require(
        !dot_reference(
            excessive_weights,
            excessive_activations,
            4,
            12,
            &rejected
        )
            && rejected.exact_sum == INT64_C(123)
            && rejected.absolute_product_sum == UINT64_C(456)
            && rejected.output == INT32_C(789),
        "unsafe cancellation bound did not reject without mutation"
    );

    DotReference null_rejected{
        INT64_C(123), UINT64_C(456), INT32_C(789)
    };
    test.require(
        !dot_reference(
            nullptr,
            cancellation_activations,
            4,
            12,
            &null_rejected
        )
            && null_rejected.exact_sum == INT64_C(123),
        "null dot input did not reject without mutation"
    );
    test.require(
        !dot_reference(
            cancellation_weights,
            cancellation_activations,
            0,
            12,
            &null_rejected
        )
            && null_rejected.exact_sum == INT64_C(123),
        "empty dot did not reject without mutation"
    );
    test.require(
        !dot_reference(
            cancellation_weights,
            cancellation_activations,
            4,
            12,
            nullptr
        ),
        "null dot output was accepted"
    );

    const int32_t expected[] = {1, -2, 3, -4};
    const int32_t actual[] = {1, -2, 3, -4};
    test.require(
        verify_i32(actual, expected, 4).passed(),
        "verifier rejected exact output"
    );
    const int32_t corrupted[] = {1, -2, 9, -4};
    const VerifyVerdict corrupted_verdict =
        verify_i32(corrupted, expected, 4);
    test.require(
        !corrupted_verdict.passed()
            && corrupted_verdict.mismatches == 1
            && corrupted_verdict.first_mismatch == 2,
        "verifier failed to identify corruption"
    );
    test.require(
        !verify_i32(nullptr, expected, 4).passed(),
        "verifier accepted missing output"
    );
    test.require(
        !verify_i32(actual, expected, 0).passed(),
        "verifier accepted empty output"
    );

    return test.result();
}

}  // namespace p13_int16

#endif
