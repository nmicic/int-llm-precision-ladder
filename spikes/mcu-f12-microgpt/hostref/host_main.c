#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int mgpt_load_mem(const void *buf, size_t len);
int mgpt_generate_sample(char *out);
void mgpt_observer_reset(void);
uint64_t mgpt_observer_hash(void);
uint32_t mgpt_observer_steps_count(void);

#define SAMPLE_COUNT 20
#define SAMPLE_BYTES 16
#define FNV64_OFFSET UINT64_C(14695981039346656037)
#define FNV64_PRIME UINT64_C(1099511628211)

typedef struct {
    uint64_t sample_hash;
    uint64_t logits_hash;
    uint32_t steps;
    char samples[SAMPLE_COUNT][SAMPLE_BYTES];
} run_result_t;

static uint64_t hash_sample(uint64_t hash, const char *sample) {
    while (*sample) {
        hash ^= (uint8_t)*sample++;
        hash *= FNV64_PRIME;
    }
    hash ^= (uint8_t)'\n';
    return hash * FNV64_PRIME;
}

static int read_file(const char *path, void **buffer_out, size_t *size_out) {
    FILE *stream = fopen(path, "rb");
    if (!stream || fseek(stream, 0, SEEK_END) != 0) return -1;
    long length = ftell(stream);
    if (length <= 0 || fseek(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        return -1;
    }
    void *buffer = malloc((size_t)length);
    if (!buffer || fread(buffer, 1, (size_t)length, stream) !=
                       (size_t)length) {
        free(buffer);
        fclose(stream);
        return -1;
    }
    fclose(stream);
    *buffer_out = buffer;
    *size_out = (size_t)length;
    return 0;
}

static int run_model(const char *label, const char *path, run_result_t *result) {
    void *buffer = NULL;
    size_t size = 0;
    if (read_file(path, &buffer, &size) != 0) {
        fprintf(stderr, "cannot read %s\n", path);
        return -1;
    }
    if (mgpt_load_mem(buffer, size) != 0) {
        fprintf(stderr, "loader rejected %s\n", path);
        free(buffer);
        return -1;
    }
    memset(result, 0, sizeof(*result));
    result->sample_hash = FNV64_OFFSET;
    mgpt_observer_reset();
    for (int sample = 0; sample < SAMPLE_COUNT; sample++) {
        mgpt_generate_sample(result->samples[sample]);
        result->sample_hash = hash_sample(
            result->sample_hash, result->samples[sample]);
    }
    result->logits_hash = mgpt_observer_hash();
    result->steps = mgpt_observer_steps_count();
    printf("LANE %s sample_hash=%016llx logits_hash=%016llx steps=%u\n",
           label,
           (unsigned long long)result->sample_hash,
           (unsigned long long)result->logits_hash,
           result->steps);
    free(buffer);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s WIDE.mgw PACKED.mgwi CORRUPT.mgwi\n",
                argv[0]);
        return 2;
    }
    run_result_t wide, packed, corrupt;
    if (run_model("wide", argv[1], &wide) != 0 ||
        run_model("packed", argv[2], &packed) != 0 ||
        run_model("corrupt", argv[3], &corrupt) != 0)
        return 2;

    for (int i = 0; i < SAMPLE_COUNT; i++)
        printf("SAMPLE %02d %s\n", i + 1, wide.samples[i]);

    int positive = wide.sample_hash == packed.sample_hash &&
                   wide.logits_hash == packed.logits_hash &&
                   wide.steps == packed.steps &&
                   memcmp(wide.samples, packed.samples,
                          sizeof(wide.samples)) == 0;
    int negative = corrupt.logits_hash != packed.logits_hash;
    printf("HOST_RESULT positive=%s negative=%s\n",
           positive ? "PASS" : "FAIL", negative ? "PASS" : "FAIL");
    return positive && negative ? 0 : 1;
}
