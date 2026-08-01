#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "safetensors.h"

static float float_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int check_value(const char *label, int64_t actual, int64_t expected) {
    if (actual == expected) return 0;
    fprintf(
        stderr,
        "%s: got 0x%016" PRIx64 ", expected 0x%016" PRIx64 "\n",
        label,
        (uint64_t)actual,
        (uint64_t)expected
    );
    return 1;
}

#define CHECK(label, actual, expected) \
    do { \
        if (check_value((label), (actual), (expected)) != 0) return 1; \
    } while (0)

int main(void) {
    const int64_t one = INT64_C(1) << 48;
    const int64_t nonfinite = INT64_C(32000) << 48;

    CHECK("F16 +1", float16_to_q1648(UINT16_C(0x3c00)), one);
    CHECK("F16 -1", float16_to_q1648(UINT16_C(0xbc00)), -one);
    CHECK(
        "F16 largest in range",
        float16_to_q1648(UINT16_C(0x77ff)),
        INT64_C(32752) << 48
    );
    CHECK("F16 +32768 saturation", float16_to_q1648(UINT16_C(0x7800)), INT64_MAX);
    CHECK("F16 -32768 endpoint", float16_to_q1648(UINT16_C(0xf800)), INT64_MIN);
    CHECK("F16 positive finite overflow", float16_to_q1648(UINT16_C(0x7bff)), INT64_MAX);
    CHECK("F16 negative finite overflow", float16_to_q1648(UINT16_C(0xfbff)), INT64_MIN);
    CHECK("F16 positive infinity", float16_to_q1648(UINT16_C(0x7c00)), nonfinite);
    CHECK("F16 negative NaN", float16_to_q1648(UINT16_C(0xfe00)), -nonfinite);

    CHECK("BF16 +1", bfloat16_to_q1648(UINT16_C(0x3f80)), one);
    CHECK("BF16 -1", bfloat16_to_q1648(UINT16_C(0xbf80)), -one);
    CHECK(
        "BF16 largest in range",
        bfloat16_to_q1648(UINT16_C(0x46ff)),
        INT64_C(32640) << 48
    );
    CHECK("BF16 +32768 saturation", bfloat16_to_q1648(UINT16_C(0x4700)), INT64_MAX);
    CHECK("BF16 -32768 endpoint", bfloat16_to_q1648(UINT16_C(0xc700)), INT64_MIN);
    CHECK("BF16 positive finite overflow", bfloat16_to_q1648(UINT16_C(0x477f)), INT64_MAX);
    CHECK("BF16 negative finite overflow", bfloat16_to_q1648(UINT16_C(0xc77f)), INT64_MIN);
    CHECK("BF16 positive infinity", bfloat16_to_q1648(UINT16_C(0x7f80)), nonfinite);
    CHECK("BF16 negative NaN", bfloat16_to_q1648(UINT16_C(0xffc0)), -nonfinite);

    CHECK("F32 +1", float32_to_q1648(float_from_bits(UINT32_C(0x3f800000))), one);
    CHECK("F32 -1", float32_to_q1648(float_from_bits(UINT32_C(0xbf800000))), -one);
    CHECK(
        "F32 largest in range",
        float32_to_q1648(float_from_bits(UINT32_C(0x46ffffff))),
        (int64_t)UINT64_C(0x7fffff8000000000)
    );
    CHECK(
        "F32 +32768 saturation",
        float32_to_q1648(float_from_bits(UINT32_C(0x47000000))),
        INT64_MAX
    );
    CHECK(
        "F32 -32768 endpoint",
        float32_to_q1648(float_from_bits(UINT32_C(0xc7000000))),
        INT64_MIN
    );
    CHECK(
        "F32 positive finite overflow",
        float32_to_q1648(float_from_bits(UINT32_C(0x7f7fffff))),
        INT64_MAX
    );
    CHECK(
        "F32 negative finite overflow",
        float32_to_q1648(float_from_bits(UINT32_C(0xff7fffff))),
        INT64_MIN
    );
    CHECK(
        "F32 positive infinity",
        float32_to_q1648(float_from_bits(UINT32_C(0x7f800000))),
        nonfinite
    );
    CHECK(
        "F32 negative NaN",
        float32_to_q1648(float_from_bits(UINT32_C(0xffc00000))),
        -nonfinite
    );
    CHECK(
        "F32 positive subnormal",
        float32_to_q1648(float_from_bits(UINT32_C(0x00000001))),
        0
    );

    puts("SAFETENSORS_CONVERSION_TEST=PASS");
    return 0;
}
