#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define MGW_VERSION 1u
#define MGW_ENDIAN_TAG UINT32_C(0x01020304)
#define FP_FRACTION_BITS 48
#define IO_BUFFER_BYTES (8u * 1024u * 1024u)
#define MAX_RULES 128

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t endian_tag;
    uint32_t num_tensors;
    uint64_t index_offset;
    uint64_t data_offset;
    uint8_t reserved[32];
} mgw_header_t;

typedef struct {
    char name[64];
    uint64_t num_elements;
    uint64_t data_offset;
    uint32_t ndims;
    uint32_t shape[2];
    uint32_t reserved;
} mgw_index_entry_t;

typedef struct {
    char *pattern;
    int bits;
    uint32_t matches;
} rule_t;

typedef struct {
    mgw_index_entry_t entry;
    char name[65];
    int bits;
} tensor_plan_t;

typedef struct {
    uint64_t tensors;
    uint64_t elements;
    uint64_t changed;
    uint64_t original_zeros;
    uint64_t quantized_zeros;
    uint64_t saturations;
    uint64_t max_abs_delta;
    long double sum_abs_delta;
    int64_t scaled_min;
    int64_t scaled_max;
    int has_scaled_value;
} bit_stats_t;

_Static_assert(sizeof(mgw_header_t) == 64, "MGW header must be 64 bytes");
_Static_assert(sizeof(mgw_index_entry_t) == 96,
               "MGW index entry must be 96 bytes");

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s --default-bits N [--set GLOB=N ...] SOURCE DEST\n"
            "       %s --analyze-only --default-bits N "
            "[--set GLOB=N ...] SOURCE\n"
            "\n"
            "Round MGW Q16.48 tensor values onto coarser fractional grids.\n"
            "Rules use fnmatch(3) syntax and later matching rules win.\n"
            "DEST must not already exist. --analyze-only computes the same\n"
            "statistics without creating a rounded copy.\n",
            program,
            program);
}

static int parse_bits(const char *text, int *bits_out) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end != '\0'
        || value < 0 || value > FP_FRACTION_BITS) {
        return -1;
    }
    *bits_out = (int)value;
    return 0;
}

static int read_exact_at(int fd, void *buffer, size_t count, off_t offset) {
    uint8_t *cursor = (uint8_t *)buffer;
    while (count > 0) {
        ssize_t got = pread(fd, cursor, count, offset);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            return -1;
        }
        cursor += (size_t)got;
        count -= (size_t)got;
        offset += got;
    }
    return 0;
}

static int write_exact(int fd, const void *buffer, size_t count) {
    const uint8_t *cursor = (const uint8_t *)buffer;
    while (count > 0) {
        ssize_t put = write(fd, cursor, count);
        if (put < 0 && errno == EINTR) {
            continue;
        }
        if (put <= 0) {
            return -1;
        }
        cursor += (size_t)put;
        count -= (size_t)put;
    }
    return 0;
}

static int tensor_compare(const void *left, const void *right) {
    const tensor_plan_t *a = (const tensor_plan_t *)left;
    const tensor_plan_t *b = (const tensor_plan_t *)right;
    if (a->entry.data_offset < b->entry.data_offset) {
        return -1;
    }
    if (a->entry.data_offset > b->entry.data_offset) {
        return 1;
    }
    return 0;
}

static uint64_t unsigned_abs_delta(int64_t left, int64_t right) {
    __int128 delta = (__int128)left - (__int128)right;
    if (delta < 0) {
        delta = -delta;
    }
    return (uint64_t)delta;
}

static int64_t round_q1648(int64_t value, int bits, int *saturated) {
    int dropped = FP_FRACTION_BITS - bits;
    *saturated = 0;
    if (dropped == 0) {
        return value;
    }

    uint64_t quantum = UINT64_C(1) << dropped;
    uint64_t mask = quantum - 1;
    uint64_t half = quantum >> 1;
    int negative = value < 0;
    uint64_t magnitude = negative
        ? (uint64_t)(-(value + 1)) + 1
        : (uint64_t)value;
    uint64_t base = magnitude & ~mask;
    uint64_t remainder = magnitude & mask;
    uint64_t rounded = base;

    if (remainder >= half) {
        if (base > UINT64_MAX - quantum) {
            rounded = UINT64_MAX;
        } else {
            rounded = base + quantum;
        }
    }

    if (negative) {
        if (rounded >= (UINT64_C(1) << 63)) {
            if (rounded > (UINT64_C(1) << 63)) {
                *saturated = 1;
            }
            return INT64_MIN;
        }
        return -(int64_t)rounded;
    }
    if (rounded > (uint64_t)INT64_MAX) {
        *saturated = 1;
        return INT64_MAX;
    }
    return (int64_t)rounded;
}

static int copy_bytes(
    int source_fd,
    int destination_fd,
    uint8_t *buffer,
    uint64_t offset,
    uint64_t count
) {
    while (count > 0) {
        size_t chunk = count > IO_BUFFER_BYTES
            ? IO_BUFFER_BYTES
            : (size_t)count;
        if (read_exact_at(source_fd, buffer, chunk, (off_t)offset) != 0
            || write_exact(destination_fd, buffer, chunk) != 0) {
            return -1;
        }
        offset += chunk;
        count -= chunk;
    }
    return 0;
}

static int round_tensor(
    int source_fd,
    int destination_fd,
    int64_t *buffer,
    const tensor_plan_t *tensor,
    bit_stats_t *stats
) {
    uint64_t remaining = tensor->entry.num_elements;
    uint64_t offset = tensor->entry.data_offset;
    bit_stats_t *bit = &stats[tensor->bits];
    bit->tensors++;
    bit->elements += remaining;

    while (remaining > 0) {
        size_t capacity = IO_BUFFER_BYTES / sizeof(int64_t);
        size_t elements = remaining > capacity
            ? capacity
            : (size_t)remaining;
        size_t bytes = elements * sizeof(int64_t);
        if (read_exact_at(source_fd, buffer, bytes, (off_t)offset) != 0) {
            return -1;
        }
        for (size_t i = 0; i < elements; i++) {
            int64_t original = buffer[i];
            int saturated = 0;
            int64_t rounded = round_q1648(
                original,
                tensor->bits,
                &saturated
            );
            uint64_t delta = unsigned_abs_delta(original, rounded);
            bit->changed += rounded != original;
            bit->original_zeros += original == 0;
            bit->quantized_zeros += rounded == 0;
            bit->saturations += (uint64_t)saturated;
            if (delta > bit->max_abs_delta) {
                bit->max_abs_delta = delta;
            }
            bit->sum_abs_delta += (long double)delta;
            int dropped = FP_FRACTION_BITS - tensor->bits;
            int64_t scale = INT64_C(1) << dropped;
            int64_t scaled = rounded / scale;
            if (!bit->has_scaled_value) {
                bit->scaled_min = scaled;
                bit->scaled_max = scaled;
                bit->has_scaled_value = 1;
            } else {
                if (scaled < bit->scaled_min) {
                    bit->scaled_min = scaled;
                }
                if (scaled > bit->scaled_max) {
                    bit->scaled_max = scaled;
                }
            }
            buffer[i] = rounded;
        }
        if (destination_fd >= 0
            && write_exact(destination_fd, buffer, bytes) != 0) {
            return -1;
        }
        offset += bytes;
        remaining -= elements;
    }
    return 0;
}

int main(int argc, char **argv) {
    int default_bits = -1;
    int analyze_only = 0;
    rule_t rules[MAX_RULES];
    size_t rule_count = 0;
    int arg = 1;
    memset(rules, 0, sizeof(rules));

    while (arg < argc && argv[arg][0] == '-') {
        if (strcmp(argv[arg], "--default-bits") == 0) {
            if (++arg >= argc || parse_bits(argv[arg], &default_bits) != 0) {
                fprintf(stderr, "invalid --default-bits value\n");
                return 2;
            }
            arg++;
        } else if (strcmp(argv[arg], "--set") == 0) {
            if (++arg >= argc || rule_count == MAX_RULES) {
                fprintf(stderr, "invalid or excessive --set rules\n");
                return 2;
            }
            char *copy = strdup(argv[arg]);
            char *equals = copy ? strrchr(copy, '=') : NULL;
            int bits = -1;
            if (!copy || !equals || equals == copy
                || parse_bits(equals + 1, &bits) != 0) {
                fprintf(stderr, "invalid --set rule: %s\n", argv[arg]);
                free(copy);
                return 2;
            }
            *equals = '\0';
            rules[rule_count].pattern = copy;
            rules[rule_count].bits = bits;
            rule_count++;
            arg++;
        } else if (strcmp(argv[arg], "--analyze-only") == 0) {
            analyze_only = 1;
            arg++;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[arg]);
            usage(argv[0]);
            return 2;
        }
    }

    int positional_count = analyze_only ? 1 : 2;
    if (default_bits < 0 || argc - arg != positional_count) {
        usage(argv[0]);
        return 2;
    }
    const char *source_path = argv[arg];
    const char *destination_path = analyze_only ? NULL : argv[arg + 1];

    int source_fd = -1;
    int destination_fd = -1;
    int exit_code = 1;
    int destination_created = 0;
    tensor_plan_t *tensors = NULL;
    uint8_t *buffer = NULL;

    source_fd = open(source_path, O_RDONLY);
    if (source_fd < 0) {
        fprintf(stderr, "cannot open source %s: %s\n",
                source_path, strerror(errno));
        goto cleanup;
    }
    struct stat source_stat;
    if (fstat(source_fd, &source_stat) != 0 || source_stat.st_size < 128) {
        fprintf(stderr, "cannot stat source or source too small\n");
        goto cleanup;
    }
    uint64_t file_size = (uint64_t)source_stat.st_size;

    mgw_header_t header;
    if (read_exact_at(source_fd, &header, sizeof(header), 0) != 0
        || memcmp(header.magic, "MGW\0", 4) != 0
        || header.version != MGW_VERSION
        || header.endian_tag != MGW_ENDIAN_TAG) {
        fprintf(stderr, "source is not host-native MGW v1\n");
        goto cleanup;
    }
    if (header.num_tensors == 0 || header.num_tensors > 1000000u) {
        fprintf(stderr, "invalid tensor count\n");
        goto cleanup;
    }
    uint64_t index_bytes =
        (uint64_t)header.num_tensors * sizeof(mgw_index_entry_t);
    if (header.index_offset > file_size
        || index_bytes > file_size - header.index_offset) {
        fprintf(stderr, "tensor index is outside the source file\n");
        goto cleanup;
    }

    tensors = (tensor_plan_t *)calloc(
        header.num_tensors,
        sizeof(tensor_plan_t)
    );
    if (!tensors) {
        fprintf(stderr, "out of memory allocating tensor plans\n");
        goto cleanup;
    }
    for (uint32_t i = 0; i < header.num_tensors; i++) {
        uint64_t entry_offset = header.index_offset
            + (uint64_t)i * sizeof(mgw_index_entry_t);
        if (read_exact_at(
                source_fd,
                &tensors[i].entry,
                sizeof(mgw_index_entry_t),
                (off_t)entry_offset
            ) != 0) {
            fprintf(stderr, "cannot read tensor index entry %u\n", i);
            goto cleanup;
        }
        const void *terminator = memchr(tensors[i].entry.name, '\0', 64);
        if (!terminator || tensors[i].entry.name[0] == '\0') {
            fprintf(stderr, "tensor %u has an invalid name\n", i);
            goto cleanup;
        }
        memcpy(tensors[i].name, tensors[i].entry.name, 64);
        tensors[i].name[64] = '\0';
        if (tensors[i].entry.ndims < 1 || tensors[i].entry.ndims > 2
            || tensors[i].entry.num_elements > UINT64_MAX / sizeof(int64_t)) {
            fprintf(stderr, "tensor %s has invalid dimensions/count\n",
                    tensors[i].name);
            goto cleanup;
        }
        uint64_t bytes =
            tensors[i].entry.num_elements * sizeof(int64_t);
        if (tensors[i].entry.data_offset > file_size
            || bytes > file_size - tensors[i].entry.data_offset
            || tensors[i].entry.data_offset % sizeof(int64_t) != 0) {
            fprintf(stderr, "tensor %s has invalid data bounds/alignment\n",
                    tensors[i].name);
            goto cleanup;
        }
        tensors[i].bits = default_bits;
        for (size_t rule_index = 0;
             rule_index < rule_count;
             rule_index++) {
            if (fnmatch(
                    rules[rule_index].pattern,
                    tensors[i].name,
                    0
                ) == 0) {
                tensors[i].bits = rules[rule_index].bits;
                rules[rule_index].matches++;
            }
        }
        for (uint32_t earlier = 0; earlier < i; earlier++) {
            if (strcmp(tensors[earlier].name, tensors[i].name) == 0) {
                fprintf(stderr, "duplicate tensor name: %s\n",
                        tensors[i].name);
                goto cleanup;
            }
        }
    }
    for (size_t i = 0; i < rule_count; i++) {
        if (rules[i].matches == 0) {
            fprintf(stderr, "rule matched no tensors: %s\n",
                    rules[i].pattern);
            goto cleanup;
        }
    }

    qsort(
        tensors,
        header.num_tensors,
        sizeof(tensor_plan_t),
        tensor_compare
    );
    uint64_t previous_end = 0;
    for (uint32_t i = 0; i < header.num_tensors; i++) {
        uint64_t start = tensors[i].entry.data_offset;
        uint64_t bytes =
            tensors[i].entry.num_elements * sizeof(int64_t);
        if (start < previous_end) {
            fprintf(stderr, "overlapping tensor data at %s\n",
                    tensors[i].name);
            goto cleanup;
        }
        previous_end = start + bytes;
    }

    if (!analyze_only) {
        destination_fd = open(
            destination_path,
            O_WRONLY | O_CREAT | O_EXCL,
            0644
        );
        if (destination_fd < 0) {
            fprintf(stderr, "cannot create destination %s: %s\n",
                    destination_path, strerror(errno));
            goto cleanup;
        }
        destination_created = 1;
    }
    buffer = (uint8_t *)malloc(IO_BUFFER_BYTES);
    if (!buffer) {
        fprintf(stderr, "out of memory allocating I/O buffer\n");
        goto cleanup;
    }

    bit_stats_t stats[FP_FRACTION_BITS + 1];
    memset(stats, 0, sizeof(stats));
    uint64_t cursor = 0;
    for (uint32_t i = 0; i < header.num_tensors; i++) {
        uint64_t start = tensors[i].entry.data_offset;
        if (!analyze_only
            && copy_bytes(
                    source_fd,
                    destination_fd,
                    buffer,
                    cursor,
                    start - cursor
                ) != 0) {
            fprintf(stderr, "copy failed before tensor %s: %s\n",
                    tensors[i].name, strerror(errno));
            goto cleanup;
        }
        if (round_tensor(
                source_fd,
                destination_fd,
                (int64_t *)buffer,
                &tensors[i],
                stats
            ) != 0) {
            fprintf(stderr, "rounding failed for tensor %s: %s\n",
                    tensors[i].name, strerror(errno));
            goto cleanup;
        }
        cursor = start
            + tensors[i].entry.num_elements * sizeof(int64_t);
    }
    if (!analyze_only
        && copy_bytes(
                source_fd,
                destination_fd,
                buffer,
                cursor,
                file_size - cursor
            ) != 0) {
        fprintf(stderr, "trailing copy failed: %s\n", strerror(errno));
        goto cleanup;
    }
    if (!analyze_only && fsync(destination_fd) != 0) {
        fprintf(stderr, "fsync failed: %s\n", strerror(errno));
        goto cleanup;
    }

    printf("source=%s\n", source_path);
    if (analyze_only) {
        printf("mode=analyze-only\n");
    } else {
        printf("destination=%s\n", destination_path);
    }
    printf("bytes=%" PRIu64 "\n", file_size);
    printf("tensors=%u\n", header.num_tensors);
    for (int bits = FP_FRACTION_BITS; bits >= 0; bits--) {
        bit_stats_t *item = &stats[bits];
        if (item->tensors == 0) {
            continue;
        }
        uint64_t newly_zero =
            item->quantized_zeros - item->original_zeros;
        long double mean = item->elements
            ? item->sum_abs_delta / (long double)item->elements
            : 0.0L;
        printf(
            "F%d tensors=%" PRIu64
            " elements=%" PRIu64
            " changed=%" PRIu64
            " newly_zero=%" PRIu64
            " max_delta_raw=%" PRIu64
            " mean_delta_raw=%.3Lf"
            " scaled_min=%" PRId64
            " scaled_max=%" PRId64
            " saturations=%" PRIu64 "\n",
            bits,
            item->tensors,
            item->elements,
            item->changed,
            newly_zero,
            item->max_abs_delta,
            mean,
            item->scaled_min,
            item->scaled_max,
            item->saturations
        );
    }
    exit_code = 0;

cleanup:
    if (source_fd >= 0) {
        close(source_fd);
    }
    if (destination_fd >= 0) {
        close(destination_fd);
    }
    if (exit_code != 0 && destination_created) {
        unlink(destination_path);
    }
    for (size_t i = 0; i < rule_count; i++) {
        free(rules[i].pattern);
    }
    free(tensors);
    free(buffer);
    return exit_code;
}
