#include "real_weight_math.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>

using real_weight::DotReference;
using real_weight::FloatingReference;
using real_weight::Q48_SCALE;
using real_weight::i128;

static void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static void test_bf16_conversion() {
    int64_t value = 0;
    require(real_weight::bf16_to_q48(0x0000, &value) && value == 0,
            "positive zero");
    require(real_weight::bf16_to_q48(0x8000, &value) && value == 0,
            "negative zero");
    require(real_weight::bf16_to_q48(0x3f80, &value)
                && value == static_cast<int64_t>(Q48_SCALE),
            "one");
    require(real_weight::bf16_to_q48(0xbf00, &value)
                && value == -static_cast<int64_t>(Q48_SCALE / 2),
            "negative half");
    require(real_weight::bf16_to_q48(0x4020, &value)
                && value == static_cast<int64_t>(5 * Q48_SCALE / 2),
            "two point five");
    require(real_weight::bf16_to_q48(0x0001, &value) && value == 0,
            "nonzero subnormal follows repository truncation to zero");
    require(!real_weight::bf16_is_exact_q48(0x0001),
            "nonzero subnormal is not exactly representable");
    require(real_weight::bf16_is_exact_q48(0x3f80),
            "one is exactly representable");
    require(real_weight::bf16_to_q48(0x2780, &value) && value == 1,
            "smallest positive Q48 unit");
    require(real_weight::bf16_to_q48(0xa780, &value) && value == -1,
            "smallest negative Q48 unit");
    require(real_weight::bf16_to_q48(0x2700, &value) && value == 0,
            "half-Q48 unit truncates to zero");
    require(!real_weight::bf16_is_exact_q48(0x2700),
            "half-Q48 unit is recorded as rounded");
    require(real_weight::bf16_to_q48(0x2781, &value) && value == 1,
            "fractional Q48 unit truncates toward zero");
    require(!real_weight::bf16_is_exact_q48(0x2781),
            "fractional Q48 unit is recorded as rounded");
    require(
        real_weight::bf16_to_q48(0x46ff, &value)
            && value == static_cast<int64_t>(
                static_cast<i128>(32640) * Q48_SCALE
            ),
        "largest positive BF16 below 32768"
    );
    require(!real_weight::bf16_to_q48(0x4700, &value),
            "positive 32768 rejects");
    require(
        real_weight::bf16_to_q48(0xc700, &value)
            && value == std::numeric_limits<int64_t>::min(),
        "negative 32768 maps to INT64_MIN"
    );
    require(!real_weight::bf16_to_q48(0x7f7f, &value),
            "large finite BF16 rejects before shifting");
    require(!real_weight::bf16_to_q48(0x7f80, &value),
            "infinity rejects");
    require(!real_weight::bf16_to_q48(0x7fc1, &value), "NaN rejects");

    for (uint32_t raw = 0; raw <= UINT16_MAX; ++raw) {
        int64_t production = 0;
        int64_t independent = 0;
        const bool production_ok = real_weight::bf16_to_q48(
            static_cast<uint16_t>(raw), &production
        );
        const bool independent_ok = real_weight::bf16_to_q48_independent(
            static_cast<uint16_t>(raw), &independent
        );
        require(production_ok == independent_ok,
                "exhaustive BF16 acceptance agrees with independent oracle");
        require(!production_ok || production == independent,
                "exhaustive BF16 value agrees with independent oracle");
        require(
            !real_weight::bf16_is_exact_q48(static_cast<uint16_t>(raw))
                || production_ok,
            "exact-on-grid values are accepted by repository conversion"
        );
        real_weight::Bf16Component component{};
        const bool component_ok = real_weight::bf16_component(
            static_cast<uint16_t>(raw), &component
        );
        const bool finite =
            (raw & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
        require(
            component_ok == finite,
            "BF16 component decoder accepts exactly finite encodings"
        );
        if (component_ok) {
            const uint32_t float_bits = raw << 16;
            float decoded_float = 0.0f;
            static_assert(
                sizeof(decoded_float) == sizeof(float_bits),
                "binary32 representation"
            );
            std::memcpy(
                &decoded_float, &float_bits, sizeof(decoded_float)
            );
            const long double decoded_component = std::ldexp(
                static_cast<long double>(component.significand),
                component.exponent
            );
            require(
                decoded_component
                    == static_cast<long double>(decoded_float),
                "BF16 component value matches independent binary32 decode"
            );
        }
    }

    real_weight::Bf16Component component{};
    require(
        real_weight::bf16_component(0x3f80, &component)
            && component.significand == 128
            && component.exponent == -7,
        "independent BF16 rational component for one"
    );
    require(
        real_weight::bf16_component(0xbf00, &component)
            && component.significand == -128
            && component.exponent == -8,
        "independent BF16 rational component for negative half"
    );
    require(
        real_weight::bf16_component(0x0001, &component)
            && component.significand == 1
            && component.exponent == -133,
        "independent BF16 rational component for subnormal"
    );
}

static void test_floor_shift() {
    int64_t value = 0;
    require(real_weight::floor_shift_q48(Q48_SCALE + 1, &value)
                && value == 1,
            "positive floor");
    require(real_weight::floor_shift_q48(-1, &value) && value == -1,
            "negative fraction floors to minus one");
    require(real_weight::floor_shift_q48(-Q48_SCALE, &value)
                && value == -1,
            "negative exact integer");
    require(real_weight::floor_shift_q48(-Q48_SCALE - 1, &value)
                && value == -2,
            "negative non-integral floor");
    const i128 too_large =
        (static_cast<i128>(std::numeric_limits<int64_t>::max()) + 1)
        * Q48_SCALE;
    require(!real_weight::floor_shift_q48(too_large, &value),
            "positive output overflow rejects");
}

static void test_dot_reference() {
    const int64_t input[] = {
        static_cast<int64_t>(Q48_SCALE),
        -static_cast<int64_t>(Q48_SCALE / 2),
        1,
    };
    const int64_t weights[] = {
        static_cast<int64_t>(Q48_SCALE / 4),
        static_cast<int64_t>(Q48_SCALE / 2),
        -static_cast<int64_t>(Q48_SCALE / 4096),
    };
    DotReference reference{};
    require(real_weight::dot_reference(input, weights, 3, &reference),
            "dot reference accepts bounded values");
    require(reference.q48 == -1,
            "dot reference preserves one final negative floor shift");
    require(reference.absolute_real_sum > 0.49L,
            "absolute-product sum recorded");
    require(
        real_weight::fp32_accumulation_bound(
            3, reference.absolute_real_sum
        ) > 0.0L,
        "positive FP32 bound"
    );
}

static void test_bf16_component_dot_reference() {
    const uint16_t input_bits[] = {0x3f80, 0x2700, 0xbf00};
    const uint16_t weight_bits[] = {0x3f00, 0x4000, 0x3f80};
    real_weight::Bf16Component input[3]{};
    real_weight::Bf16Component weights[3]{};
    for (std::size_t index = 0; index < 3; ++index) {
        require(
            real_weight::bf16_component(input_bits[index], &input[index]),
            "input BF16 component decodes"
        );
        require(
            real_weight::bf16_component(weight_bits[index], &weights[index]),
            "weight BF16 component decodes"
        );
    }
    FloatingReference reference{};
    require(
        real_weight::bf16_component_dot_reference(
            input, weights, 3, &reference
        ),
        "BF16 component dot reference accepts bounded values"
    );
    require(
        reference.exact_real == 0x1p-48L,
        "BF16 component reference preserves a below-grid contribution"
    );
    require(
        reference.absolute_real_sum > 1.0L,
        "BF16 component reference records absolute-product sum"
    );
}

static void test_production_verifiers() {
    DotReference integer_expected[] = {
        {11, 0.0L, 0.0L},
        {-7, 0.0L, 0.0L},
        {3, 0.0L, 0.0L},
    };
    const int64_t integer_good[] = {11, -7, 3};
    const int64_t integer_corrupt[] = {11, -6, 3};
    const int64_t integer_poison[] = {
        static_cast<int64_t>(UINT64_C(0xa5a5a5a5a5a5a5a5)),
        static_cast<int64_t>(UINT64_C(0xa5a5a5a5a5a5a5a5)),
        static_cast<int64_t>(UINT64_C(0xa5a5a5a5a5a5a5a5)),
    };
    require(
        real_weight::verify_integer_values(
            integer_good, integer_expected, 3
        ).passed(),
        "production integer verifier accepts known-good output"
    );
    require(
        !real_weight::verify_integer_values(
            integer_corrupt, integer_expected, 3
        ).passed(),
        "production integer verifier rejects corruption"
    );
    require(
        !real_weight::verify_integer_values(
            integer_poison, integer_expected, 3
        ).passed(),
        "production integer verifier rejects skipped poisoned output"
    );

    const FloatingReference floating_expected[] = {
        {1.0L, 1.0L},
        {-2.0L, 2.0L},
        {0.5L, 0.5L},
    };
    const float floating_good[] = {1.0f, -2.0f, 0.5f};
    const float floating_corrupt[] = {1.0f, -1.0f, 0.5f};
    const float floating_partial[] = {
        1.0f, -2.0f, std::numeric_limits<float>::quiet_NaN()
    };
    require(
        real_weight::verify_floating_values(
            floating_good, floating_expected, 3, 8
        ).passed(),
        "production floating verifier accepts known-good output"
    );
    require(
        !real_weight::verify_floating_values(
            floating_corrupt, floating_expected, 3, 8
        ).passed(),
        "production floating verifier rejects corruption"
    );
    require(
        !real_weight::verify_floating_values(
            floating_partial, floating_expected, 3, 8
        ).passed(),
        "production floating verifier rejects partial poisoned output"
    );

    const FloatingReference zero_expected[] = {{0.0L, 0.0L}};
    const uint32_t poison_patterns[] = {
        UINT32_C(0x7fc00001),
        UINT32_C(0x7fc00002),
        UINT32_C(0xa5a5a5a5),
    };
    for (uint32_t poison_bits : poison_patterns) {
        float poison_value = 0.0f;
        std::memcpy(&poison_value, &poison_bits, sizeof(poison_value));
        require(
            !real_weight::verify_floating_values(
                &poison_value,
                zero_expected,
                1,
                1,
                poison_bits,
                true
            ).passed(),
            "production floating verifier rejects its exact output sentinel"
        );
    }
}

static void test_environment_gate() {
    constexpr std::size_t memory = 32ULL << 30;
    require(
        real_weight::target_environment_matches(
            "NVIDIA GeForce RTX 5090", 12, 0, memory, 13030
        ),
        "pinned GPU/runtime environment accepts"
    );
    require(
        !real_weight::target_environment_matches(
            "NVIDIA GeForce RTX 5090", 12, 1, memory, 13030
        ),
        "wrong compute capability rejects"
    );
    require(
        !real_weight::target_environment_matches(
            "NVIDIA GeForce RTX 5090", 12, 0, memory, 13020
        ),
        "wrong CUDA runtime rejects"
    );
    require(
        !real_weight::target_environment_matches(
            "substituted GPU", 12, 0, memory, 13030
        ),
        "wrong device name rejects"
    );
}

int main() {
    test_bf16_conversion();
    test_floor_shift();
    test_dot_reference();
    test_bf16_component_dot_reference();
    test_production_verifiers();
    test_environment_gate();
    std::puts("real-weight host math: PASS");
    return 0;
}
