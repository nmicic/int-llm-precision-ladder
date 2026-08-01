#include <Arduino.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "pins.h"
#include "wide_model.h"
#include "packed_model.h"
#include "corrupt_model.h"

#if ((defined(MGPT_FULL_DUAL) ? 1 : 0) + \
     (defined(MGPT_FULL_WIDE_ONLY) ? 1 : 0) + \
     (defined(MGPT_FULL_PACKED_ONLY) ? 1 : 0)) != 1
#error "select exactly one full-model firmware mode"
#endif

extern "C" int mgpt_load_mem(const void *buffer, size_t length);
extern "C" int mgpt_generate_sample(char *output);
extern "C" void mgpt_observer_reset(void);
extern "C" uint64_t mgpt_observer_hash(void);
extern "C" uint32_t mgpt_observer_steps_count(void);

#define SAMPLE_COUNT 20
#define FNV64_OFFSET UINT64_C(14695981039346656037)
#define FNV64_PRIME UINT64_C(1099511628211)

struct LaneResult {
    uint64_t sample_hash;
    uint64_t logits_hash;
    uint32_t steps;
    uint32_t elapsed_us;
    bool loaded;
};

static uint64_t hash_sample(uint64_t hash, const char *sample) {
    while (*sample) {
        hash ^= (uint8_t)*sample++;
        hash *= FNV64_PRIME;
    }
    hash ^= (uint8_t)'\n';
    return hash * FNV64_PRIME;
}

static LaneResult run_lane(const uint8_t *model, size_t length) {
    LaneResult result = {FNV64_OFFSET, 0, 0, 0, false};
    if (mgpt_load_mem(model, length) != 0)
        return result;
    result.loaded = true;
    mgpt_observer_reset();
    char sample[16];
    uint32_t start = micros();
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        mgpt_generate_sample(sample);
        result.sample_hash = hash_sample(result.sample_hash, sample);
    }
    result.elapsed_us = micros() - start;
    result.logits_hash = mgpt_observer_hash();
    result.steps = mgpt_observer_steps_count();
    return result;
}

static bool is_positive(const LaneResult &result) {
    return result.loaded && result.sample_hash == PIN_SAMPLE_HASH &&
           result.logits_hash == PIN_LOGITS_HASH &&
           result.steps == PIN_STEPS;
}

static void print_hex64(uint64_t value) {
    char text[17];
    for (int i = 15; i >= 0; i--) {
        text[i] = "0123456789abcdef"[value & 15];
        value >>= 4;
    }
    text[16] = 0;
    Serial.print(text);
}

static void print_call(int index, char lane, const LaneResult &result,
                       bool pass) {
    Serial.print("CALL index="); Serial.print(index);
    Serial.print(" lane="); Serial.print(lane);
    Serial.print(" us="); Serial.print(result.elapsed_us);
    Serial.print(" sample_hash="); print_hex64(result.sample_hash);
    Serial.print(" logits_hash="); print_hex64(result.logits_hash);
    Serial.print(" steps="); Serial.print(result.steps);
    Serial.print(" status="); Serial.println(pass ? "PASS" : "FAIL");
}

static void run_once(void) {
    bool correctness = true;
    bool negative = true;
    Serial.println("MICROGPT_F12 schema=full_model_v1 board=xiao-rp2040 arch=cortex-m0plus clock_hz=133000000");
    Serial.print("wide_bytes="); Serial.println(WIDE_MODEL_BYTES);
    Serial.print("packed_bytes="); Serial.println(PACKED_MODEL_BYTES);
    Serial.print("wide_sha256="); Serial.println(WIDE_MODEL_SHA256);
    Serial.print("packed_sha256="); Serial.println(PACKED_MODEL_SHA256);

#if defined(MGPT_FULL_DUAL)
    LaneResult warm_wide = run_lane(wide_model, wide_model_len);
    LaneResult warm_packed = run_lane(packed_model, packed_model_len);
    bool warm_ok = is_positive(warm_wide) && is_positive(warm_packed);
    Serial.print("WARMUP status="); Serial.println(warm_ok ? "PASS" : "FAIL");
    correctness &= warm_ok;

    const char schedule[] = {'W', 'P', 'P', 'W', 'P', 'W', 'W', 'P'};
    for (int i = 0; i < (int)sizeof(schedule); i++) {
        LaneResult result = schedule[i] == 'W'
            ? run_lane(wide_model, wide_model_len)
            : run_lane(packed_model, packed_model_len);
        bool pass = is_positive(result);
        correctness &= pass;
        print_call(i + 1, schedule[i], result, pass);
    }
    LaneResult corrupt = run_lane(corrupt_model, corrupt_model_len);
    negative = corrupt.loaded &&
               corrupt.sample_hash == PIN_CORRUPT_SAMPLE_HASH &&
               corrupt.logits_hash == PIN_CORRUPT_LOGITS_HASH &&
               corrupt.steps == PIN_CORRUPT_STEPS &&
               corrupt.logits_hash != PIN_LOGITS_HASH;
    Serial.print("NEGATIVE sample_hash="); print_hex64(corrupt.sample_hash);
    Serial.print(" logits_hash="); print_hex64(corrupt.logits_hash);
    Serial.print(" steps="); Serial.print(corrupt.steps);
    Serial.print(" status="); Serial.println(negative ? "PASS" : "FAIL");
#elif defined(MGPT_FULL_WIDE_ONLY)
    LaneResult result = run_lane(wide_model, wide_model_len);
    correctness = is_positive(result);
    print_call(1, 'W', result, correctness);
#else
    LaneResult result = run_lane(packed_model, packed_model_len);
    correctness = is_positive(result);
    print_call(1, 'P', result, correctness);
#endif

    Serial.print("MICROGPT_F12_RESULT correctness=");
    Serial.print(correctness ? "PASS" : "FAIL");
    Serial.print(" negative="); Serial.print(negative ? "PASS" : "FAIL");
    Serial.print(" sample_hash="); print_hex64(PIN_SAMPLE_HASH);
    Serial.print(" logits_hash="); print_hex64(PIN_LOGITS_HASH);
    Serial.print(" steps="); Serial.println(PIN_STEPS);
    Serial.println("DONE");
}

static char command[64];
static size_t command_length = 0;
static bool completed = false;

void setup(void) {
    Serial.begin(115200);
}

void loop(void) {
    if (completed) return;
    while (Serial.available() > 0) {
        int byte = Serial.read();
        if (byte == '\r') continue;
        if (byte == '\n') {
            command[command_length] = 0;
            if (command_length == 36 && memcmp(command, "RUN ", 4) == 0) {
                Serial.print("BEGIN challenge=");
                Serial.println(command + 4);
                run_once();
                completed = true;
            } else {
                Serial.println("INPUT_FAIL");
            }
            command_length = 0;
        } else if (command_length + 1 < sizeof(command)) {
            command[command_length++] = (char)byte;
        } else {
            command_length = 0;
        }
    }
}
