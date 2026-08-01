/*
 * Final bounded P13 GPU follow-up.
 *
 * This translation unit imports the reviewed OPEN-S5 arithmetic, kernels, and
 * self-tests without editing them. Its new measurement unit is a sequence of
 * 12 GEMVs over 12 distinct resident weight addresses.
 */

#define main p13_archived_single_launch_main
#include "../gpu-p13-next/bench_p13_int16.cu"
#undef main

#ifndef P13_ROTATED_SOURCE_SHA256
#define P13_ROTATED_SOURCE_SHA256 "UNLOCKED"
#endif
#ifndef P13_ROTATED_PROTOCOL_SHA256
#define P13_ROTATED_PROTOCOL_SHA256 "UNLOCKED"
#endif
#ifndef P13_ROTATED_BUILD_ID
#define P13_ROTATED_BUILD_ID "UNLOCKED"
#endif

namespace {

constexpr std::size_t ROTATION_COPIES = 12;
constexpr uint64_t ROTATED_PRESSURE_INITIAL =
    UINT64_C(0x3535353535353535);
constexpr uint64_t ROTATED_PRESSURE_MULTIPLIER =
    UINT64_C(6364136223846793005);
constexpr uint64_t ROTATED_PRESSURE_INCREMENT =
    UINT64_C(1442695040888963407);
constexpr uint64_t ROTATED_FNV_OFFSET =
    UINT64_C(1469598103934665603);
constexpr uint64_t ROTATED_FNV_PRIME =
    UINT64_C(1099511628211);
constexpr Shape ROTATED_SHAPES[] = {
    {"gate_up", 5632, 2048},
    {"down_proj", 2048, 5632},
};

constexpr std::array<std::array<Lane, 3>, 6> ROTATED_PERMUTATIONS{{
    {{Lane::FP32, Lane::WIDE, Lane::PACKED}},
    {{Lane::FP32, Lane::PACKED, Lane::WIDE}},
    {{Lane::WIDE, Lane::FP32, Lane::PACKED}},
    {{Lane::WIDE, Lane::PACKED, Lane::FP32}},
    {{Lane::PACKED, Lane::FP32, Lane::WIDE}},
    {{Lane::PACKED, Lane::WIDE, Lane::FP32}},
}};

enum class RotatedMode {
    RETAINED,
    QUALIFY_ONLY,
    SANITIZE_ONLY,
};

struct RotatedEvidenceCounts {
    std::size_t boundaries = 0;
    std::size_t preflights = 0;
    std::size_t samples = 0;
    std::size_t summaries = 0;
    std::size_t posts = 0;
    std::size_t shapes = 0;
};

std::size_t checked_product(std::size_t left, std::size_t right) {
    if (left == 0
        || right == 0
        || left > std::numeric_limits<std::size_t>::max() / right) {
        fail("rotated allocation element-count overflow");
    }
    return left * right;
}

uint64_t rounded_divide(uint64_t numerator, uint64_t denominator) {
    if (denominator == 0) fail("zero rounded-division denominator");
    uint64_t quotient = numerator / denominator;
    const uint64_t remainder = numerator % denominator;
    if (remainder >= denominator - remainder) {
        if (quotient == std::numeric_limits<uint64_t>::max()) {
            fail("rounded-division overflow");
        }
        ++quotient;
    }
    return quotient;
}

uint64_t event_ticks(float milliseconds) {
    if (!std::isfinite(milliseconds) || milliseconds <= 0.0f) {
        fail("invalid rotated CUDA event duration");
    }
    const long double scaled =
        static_cast<long double>(milliseconds) * 1000000000.0L;
    if (!std::isfinite(scaled)
        || scaled < 1.0L
        || scaled
            > static_cast<long double>(
                std::numeric_limits<uint64_t>::max()
            ) - 0.5L) {
        fail("rotated CUDA event duration does not fit ticks");
    }
    return static_cast<uint64_t>(std::floor(scaled + 0.5L));
}

uint64_t median_twice(std::vector<uint64_t> values) {
    if (values.empty()) fail("median of empty rotated timing vector");
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0) {
        if (values[middle] > std::numeric_limits<uint64_t>::max() / 2) {
            fail("rotated timing median overflow");
        }
        return values[middle] * 2;
    }
    if (values[middle - 1]
        > std::numeric_limits<uint64_t>::max() - values[middle]) {
        fail("rotated timing median overflow");
    }
    return values[middle - 1] + values[middle];
}

void print_ticks(uint64_t value) {
    std::printf(
        "%llu.%06llu",
        static_cast<unsigned long long>(value / 1000000),
        static_cast<unsigned long long>(value % 1000000)
    );
}

void require_rotated_build_identity() {
    require_locked_build_identity();
    if (std::strcmp(P13_ROTATED_SOURCE_SHA256, "UNLOCKED") == 0
        || std::strcmp(P13_ROTATED_PROTOCOL_SHA256, "UNLOCKED") == 0
        || std::strcmp(P13_ROTATED_BUILD_ID, "UNLOCKED") == 0) {
        fail("rotated benchmark was not built by a hash-bound recipe");
    }
}

RotatedMode parse_rotated_mode(int argc, char **argv) {
    if (argc == 1) return RotatedMode::RETAINED;
    if (argc == 2
        && std::strcmp(argv[1], "--qualify-only") == 0) {
        return RotatedMode::QUALIFY_ONLY;
    }
    if (argc == 2
        && std::strcmp(argv[1], "--sanitize-only") == 0) {
        return RotatedMode::SANITIZE_ONLY;
    }
    fail(
        "usage: bench_p13_rotated "
        "[--qualify-only|--sanitize-only]"
    );
}

const char *rotated_mode_name(RotatedMode mode) {
    switch (mode) {
        case RotatedMode::RETAINED: return "retained";
        case RotatedMode::QUALIFY_ONLY: return "qualify_only";
        case RotatedMode::SANITIZE_ONLY: return "sanitize_only";
    }
    fail("unknown rotated mode");
}

struct PressureVerification {
    uint64_t generation = 0;
    std::size_t verified_words = 0;
    uint64_t checksum = 0;
};

class RotatedPressureVerifier {
public:
    explicit RotatedPressureVerifier(std::size_t count)
        : observed_(count) {
        if (count == 0) fail("empty rotated pressure verifier");
    }

    PressureVerification advance_and_verify(
        DeviceBuffer<uint64_t> &pressure
    ) {
        if (pressure.count() != observed_.size()) {
            fail("rotated pressure verifier size mismatch");
        }
        if (generation_ == std::numeric_limits<uint64_t>::max()) {
            fail("rotated pressure generation overflow");
        }
        apply_pressure(pressure);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            observed_.data(),
            pressure.get(),
            pressure.bytes(),
            cudaMemcpyDeviceToHost
        ));
        ++generation_;
        multiplier_coefficient_ *= ROTATED_PRESSURE_MULTIPLIER;
        additive_coefficient_ =
            additive_coefficient_ * ROTATED_PRESSURE_MULTIPLIER + 1;
        uint64_t checksum = ROTATED_FNV_OFFSET;
        for (std::size_t index = 0; index < observed_.size(); ++index) {
            const uint64_t expected =
                multiplier_coefficient_ * ROTATED_PRESSURE_INITIAL
                + additive_coefficient_
                    * (
                        static_cast<uint64_t>(index)
                        + ROTATED_PRESSURE_INCREMENT
                    );
            if (observed_[index] != expected) {
                fail(
                    std::string("rotated pressure mismatch at word ")
                    + std::to_string(index)
                );
            }
            checksum ^= observed_[index];
            checksum *= ROTATED_FNV_PRIME;
        }
        return {
            generation_,
            observed_.size(),
            checksum,
        };
    }

private:
    std::vector<uint64_t> observed_;
    uint64_t generation_ = 0;
    uint64_t multiplier_coefficient_ = 1;
    uint64_t additive_coefficient_ = 0;
};

void run_rotated_pressure_selftest() {
    constexpr std::size_t WORDS = 257;
    DeviceBuffer<uint64_t> pressure(WORDS);
    CUDA_CHECK(cudaMemset(
        pressure.get(), 0x35, pressure.bytes()
    ));
    RotatedPressureVerifier verifier(WORDS);
    const PressureVerification first =
        verifier.advance_and_verify(pressure);
    const PressureVerification second =
        verifier.advance_and_verify(pressure);
    if (first.generation != 1
        || second.generation != 2
        || first.verified_words != WORDS
        || second.verified_words != WORDS
        || first.checksum == second.checksum) {
        fail("rotated pressure verifier self-test failed");
    }
    std::printf(
        "PRESSURE_SELFTEST generations=2 verified_words=%zu "
        "checksums_distinct=PASS\n",
        WORDS
    );
}

template <typename T>
double upload_rotated_images(
    DeviceBuffer<T> &destination,
    const std::vector<T> &source
) {
    if (source.empty()
        || destination.count()
            != checked_product(source.size(), ROTATION_COPIES)) {
        fail("rotated upload size mismatch");
    }
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t copy = 0; copy < ROTATION_COPIES; ++copy) {
        CUDA_CHECK(cudaMemcpy(
            destination.get() + copy * source.size(),
            source.data(),
            source.size() * sizeof(T),
            cudaMemcpyHostToDevice
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    const double microseconds =
        std::chrono::duration<double, std::micro>(
            stop - start
        ).count();
    if (!std::isfinite(microseconds) || microseconds <= 0.0) {
        fail("invalid rotated H2D duration");
    }
    return microseconds;
}

struct RotatedDeviceFixture {
    explicit RotatedDeviceFixture(const Fixture &fixture)
        : image_elements(
              checked_elements(
                  fixture.shape.outputs,
                  fixture.shape.inputs
              )
          ),
          output_elements(
              static_cast<std::size_t>(fixture.shape.outputs)
          ),
          weights_wide(
              checked_product(image_elements, ROTATION_COPIES)
          ),
          weights_packed(
              checked_product(image_elements, ROTATION_COPIES)
          ),
          weights_fp32(
              checked_product(image_elements, ROTATION_COPIES)
          ),
          activations_i32(fixture.activations_i32.size()),
          activations_fp32(fixture.activations_fp32.size()),
          outputs_i32(
              checked_product(output_elements, ROTATION_COPIES)
          ),
          outputs_fp32(
              checked_product(output_elements, ROTATION_COPIES)
          ) {
        if (fixture.weights_wide.size() != image_elements
            || fixture.weights_packed.size() != image_elements
            || fixture.weights_fp32.size() != image_elements
            || fixture.expected_i32.size() != output_elements
            || fixture.expected_fp32.size() != output_elements) {
            fail("rotated fixture shape mismatch");
        }
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
        wide_h2d_us = upload_rotated_images(
            weights_wide, fixture.weights_wide
        );
        packed_h2d_us = upload_rotated_images(
            weights_packed, fixture.weights_packed
        );
        fp32_h2d_us = upload_rotated_images(
            weights_fp32, fixture.weights_fp32
        );
    }

    const int32_t *wide_image(std::size_t copy) const {
        if (copy >= ROTATION_COPIES) fail("wide copy index overflow");
        return weights_wide.get() + copy * image_elements;
    }

    const int16_t *packed_image(std::size_t copy) const {
        if (copy >= ROTATION_COPIES) fail("packed copy index overflow");
        return weights_packed.get() + copy * image_elements;
    }

    const float *fp32_image(std::size_t copy) const {
        if (copy >= ROTATION_COPIES) fail("FP32 copy index overflow");
        return weights_fp32.get() + copy * image_elements;
    }

    int32_t *integer_output(std::size_t copy) {
        if (copy >= ROTATION_COPIES) {
            fail("integer output copy index overflow");
        }
        return outputs_i32.get() + copy * output_elements;
    }

    float *fp32_output(std::size_t copy) {
        if (copy >= ROTATION_COPIES) {
            fail("FP32 output copy index overflow");
        }
        return outputs_fp32.get() + copy * output_elements;
    }

    std::size_t image_elements;
    std::size_t output_elements;
    DeviceBuffer<int32_t> weights_wide;
    DeviceBuffer<int16_t> weights_packed;
    DeviceBuffer<float> weights_fp32;
    DeviceBuffer<int32_t> activations_i32;
    DeviceBuffer<float> activations_fp32;
    DeviceBuffer<int32_t> outputs_i32;
    DeviceBuffer<float> outputs_fp32;
    double wide_h2d_us = 0.0;
    double packed_h2d_us = 0.0;
    double fp32_h2d_us = 0.0;
};

void poison_rotated_outputs(
    Lane lane,
    RotatedDeviceFixture &device,
    int observation
) {
    if (lane == Lane::FP32) {
        const uint32_t poison =
            (observation & 1) == 0 ? FP_POISON_A : FP_POISON_B;
        fill_fp32_bits<<<
            grid_for_count(device.outputs_fp32.count()), 256
        >>>(
            device.outputs_fp32.get(),
            device.outputs_fp32.count(),
            poison
        );
        check_launch("rotated FP32 poison");
        return;
    }
    const int32_t poison =
        (observation & 1) == 0 ? INT_POISON_A : INT_POISON_B;
    fill_i32<<<grid_for_count(device.outputs_i32.count()), 256>>>(
        device.outputs_i32.get(),
        device.outputs_i32.count(),
        poison
    );
    check_launch("rotated integer poison");
}

void enqueue_rotated_lane(
    Lane lane,
    const Fixture &fixture,
    RotatedDeviceFixture &device,
    std::size_t copies
) {
    if (copies == 0 || copies > ROTATION_COPIES) {
        fail("invalid rotated enqueue copy count");
    }
    for (std::size_t copy = 0; copy < copies; ++copy) {
        switch (lane) {
            case Lane::FP32:
                launch_fp32(
                    device.fp32_output(copy),
                    device.fp32_image(copy),
                    device.activations_fp32.get(),
                    fixture.shape.outputs,
                    fixture.shape.inputs
                );
                break;
            case Lane::WIDE:
                launch_wide(
                    device.integer_output(copy),
                    device.wide_image(copy),
                    device.activations_i32.get(),
                    fixture.shape.outputs,
                    fixture.shape.inputs
                );
                break;
            case Lane::PACKED:
                launch_packed(
                    device.integer_output(copy),
                    device.packed_image(copy),
                    device.activations_i32.get(),
                    fixture.shape.outputs,
                    fixture.shape.inputs
                );
                break;
        }
    }
}

bool rotated_integer_outputs_match(
    RotatedDeviceFixture &device,
    const std::vector<int32_t> &expected
) {
    if (expected.size() != device.output_elements) {
        fail("rotated integer expected size mismatch");
    }
    std::vector<int32_t> observed(device.outputs_i32.count());
    CUDA_CHECK(cudaMemcpy(
        observed.data(),
        device.outputs_i32.get(),
        device.outputs_i32.bytes(),
        cudaMemcpyDeviceToHost
    ));
    for (std::size_t copy = 0; copy < ROTATION_COPIES; ++copy) {
        const p13_int16::VerifyVerdict verdict =
            p13_int16::verify_i32(
                observed.data() + copy * device.output_elements,
                expected.data(),
                expected.size()
            );
        if (!verdict.passed()) return false;
    }
    return true;
}

FloatingVerdict verify_rotated_fp32_outputs(
    RotatedDeviceFixture &device,
    const std::vector<FloatingReference> &expected,
    uint32_t poison
) {
    if (expected.size() != device.output_elements) {
        fail("rotated FP32 expected size mismatch");
    }
    std::vector<float> observed(device.outputs_fp32.count());
    std::vector<float> slice(expected.size());
    CUDA_CHECK(cudaMemcpy(
        observed.data(),
        device.outputs_fp32.get(),
        device.outputs_fp32.bytes(),
        cudaMemcpyDeviceToHost
    ));
    FloatingVerdict aggregate;
    for (std::size_t copy = 0; copy < ROTATION_COPIES; ++copy) {
        std::copy(
            observed.begin()
                + static_cast<std::ptrdiff_t>(
                    copy * device.output_elements
                ),
            observed.begin()
                + static_cast<std::ptrdiff_t>(
                    (copy + 1) * device.output_elements
                ),
            slice.begin()
        );
        aggregate.merge(verify_fp32(slice, expected, poison));
    }
    if (!aggregate.passed()
        || aggregate.calls != ROTATION_COPIES
        || aggregate.elements
            != expected.size() * ROTATION_COPIES) {
        fail("rotated FP32 verification failed");
    }
    return aggregate;
}

FloatingVerdict verify_rotated_lane(
    Lane lane,
    const Fixture &fixture,
    RotatedDeviceFixture &device,
    int observation
) {
    if (lane == Lane::FP32) {
        const uint32_t poison =
            (observation & 1) == 0 ? FP_POISON_A : FP_POISON_B;
        return verify_rotated_fp32_outputs(
            device, fixture.expected_fp32, poison
        );
    }
    if (!rotated_integer_outputs_match(
            device, fixture.expected_i32
        )) {
        fail("rotated integer output verification failed");
    }
    return {};
}

void run_rotated_untimed(
    Lane lane,
    const Fixture &fixture,
    RotatedDeviceFixture &device,
    DeviceBuffer<uint64_t> *pressure,
    int observation,
    FloatingVerdict *floating
) {
    if (floating == nullptr) fail("missing rotated FP32 aggregate");
    if (pressure != nullptr) apply_pressure(*pressure);
    poison_rotated_outputs(lane, device, observation);
    enqueue_rotated_lane(
        lane, fixture, device, ROTATION_COPIES
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    const FloatingVerdict verdict =
        verify_rotated_lane(lane, fixture, device, observation);
    if (lane == Lane::FP32) floating->merge(verdict);
}

struct RotatedTimedResult {
    uint64_t sequence_ticks = 0;
    PressureVerification pressure;
};

RotatedTimedResult run_rotated_timed(
    Lane lane,
    const Fixture &fixture,
    RotatedDeviceFixture &device,
    DeviceBuffer<uint64_t> &pressure,
    RotatedPressureVerifier &pressure_verifier,
    int observation,
    FloatingVerdict *floating,
    EventPair &events
) {
    if (floating == nullptr) fail("missing timed FP32 aggregate");
    const PressureVerification pressure_result =
        pressure_verifier.advance_and_verify(pressure);
    poison_rotated_outputs(lane, device, observation);
    CUDA_CHECK(cudaEventRecord(events.start()));
    enqueue_rotated_lane(
        lane, fixture, device, ROTATION_COPIES
    );
    CUDA_CHECK(cudaEventRecord(events.stop()));
    CUDA_CHECK(cudaEventSynchronize(events.stop()));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds, events.start(), events.stop()
    ));
    const FloatingVerdict verdict =
        verify_rotated_lane(lane, fixture, device, observation);
    if (lane == Lane::FP32) floating->merge(verdict);
    return {
        event_ticks(milliseconds),
        pressure_result,
    };
}

void run_rotated_negative_control() {
    const Shape shape{"rotated_boundary_n33_k257", 33, 257};
    Fixture fixture = make_fixture(shape);
    RotatedDeviceFixture device(fixture);
    require_integer_poison_distinct(
        fixture.expected_i32, INT_POISON_A
    );
    require_integer_poison_distinct(
        fixture.expected_i32, INT_POISON_B
    );

    FloatingVerdict floating;
    for (const Lane lane : {
            Lane::FP32, Lane::WIDE, Lane::PACKED
        }) {
        run_rotated_untimed(
            lane, fixture, device, nullptr, 0, &floating
        );
    }

    poison_rotated_outputs(Lane::PACKED, device, 1);
    enqueue_rotated_lane(
        Lane::PACKED,
        fixture,
        device,
        ROTATION_COPIES - 1
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    if (rotated_integer_outputs_match(
            device, fixture.expected_i32
        )) {
        fail("rotated verifier accepted one skipped output slice");
    }
    if (!floating.passed()
        || floating.calls != ROTATION_COPIES
        || floating.elements
            != fixture.expected_fp32.size() * ROTATION_COPIES) {
        fail("rotated boundary FP32 aggregate is invalid");
    }
    std::printf(
        "ROTATED_SELFTEST copies=%zu pointer_offsets=PASS "
        "wide_outputs=PASS packed_outputs=PASS fp32_outputs=PASS "
        "skipped_slice_negative=PASS\n",
        ROTATION_COPIES
    );
}

void require_rotated_working_set(
    const RotatedDeviceFixture &device,
    std::size_t l2_bytes
) {
    if (l2_bytes == 0
        || l2_bytes > std::numeric_limits<std::size_t>::max() / 2) {
        fail("invalid L2 size for rotated admission");
    }
    const std::size_t minimum = l2_bytes * 2;
    if (device.weights_wide.bytes() < minimum
        || device.weights_packed.bytes() < minimum
        || device.weights_fp32.bytes() < minimum) {
        fail("one rotated lane is below 2x reported L2");
    }
}

struct RotatedSummary {
    uint64_t fp32_sequence_twice_ticks = 0;
    uint64_t wide_sequence_twice_ticks = 0;
    uint64_t packed_sequence_twice_ticks = 0;
};

RotatedSummary run_rotated_schedule(
    const Fixture &fixture,
    RotatedDeviceFixture &device,
    DeviceBuffer<uint64_t> &pressure,
    RotatedPressureVerifier &pressure_verifier,
    FloatingVerdict *floating,
    RotatedEvidenceCounts *evidence
) {
    if (floating == nullptr || evidence == nullptr) {
        fail("missing rotated schedule evidence state");
    }
    std::array<std::vector<uint64_t>, 3> samples;
    for (auto &lane_samples : samples) lane_samples.reserve(12);
    std::array<int, 3> observations{{0, 0, 0}};
    EventPair events;

    for (int cycle = 0; cycle < 2; ++cycle) {
        for (int step = 0; step < 6; ++step) {
            const int permutation_index =
                cycle == 0 ? step : 5 - step;
            for (int position = 0; position < 3; ++position) {
                const Lane lane =
                    ROTATED_PERMUTATIONS[
                        static_cast<std::size_t>(permutation_index)
                    ][static_cast<std::size_t>(position)];
                const std::size_t slot = lane_slot(lane);
                const int observation = observations[slot];
                const RotatedTimedResult timed = run_rotated_timed(
                    lane,
                    fixture,
                    device,
                    pressure,
                    pressure_verifier,
                    observation,
                    floating,
                    events
                );
                samples[slot].push_back(timed.sequence_ticks);
                ++observations[slot];
                std::printf(
                    "SAMPLE shape=%s regime=rotated_after_l2_pressure "
                    "cycle=%d permutation=%d position=%d lane=%s "
                    "lane_observation=%d poison=%c copies=%zu "
                    "pressure_generation=%llu "
                    "pressure_verified_words=%zu "
                    "pressure_checksum_fnv64=%016llx sequence_us=",
                    fixture.shape.id,
                    cycle,
                    permutation_index,
                    position,
                    lane_name(lane),
                    observation,
                    (observation & 1) == 0 ? 'A' : 'B',
                    ROTATION_COPIES,
                    static_cast<unsigned long long>(
                        timed.pressure.generation
                    ),
                    timed.pressure.verified_words,
                    static_cast<unsigned long long>(
                        timed.pressure.checksum
                    )
                );
                print_ticks(timed.sequence_ticks);
                std::printf(" per_launch_us=");
                print_ticks(rounded_divide(
                    timed.sequence_ticks, ROTATION_COPIES
                ));
                std::printf(
                    " outputs_verified=%zu\n",
                    ROTATION_COPIES
                );
                ++evidence->samples;
            }
        }
    }

    for (std::size_t slot = 0; slot < samples.size(); ++slot) {
        if (samples[slot].size() != 12
            || observations[slot] != 12) {
            fail("incomplete rotated timing schedule");
        }
    }
    RotatedSummary summary{
        median_twice(samples[lane_slot(Lane::FP32)]),
        median_twice(samples[lane_slot(Lane::WIDE)]),
        median_twice(samples[lane_slot(Lane::PACKED)]),
    };
    const uint64_t fp32_display = rounded_divide(
        summary.fp32_sequence_twice_ticks, 2
    );
    const uint64_t wide_display = rounded_divide(
        summary.wide_sequence_twice_ticks, 2
    );
    const uint64_t packed_display = rounded_divide(
        summary.packed_sequence_twice_ticks, 2
    );
    std::printf(
        "SUMMARY shape=%s regime=rotated_after_l2_pressure "
        "samples_per_lane=12 copies_per_sample=%zu "
        "fp32_sequence_median_us=",
        fixture.shape.id,
        ROTATION_COPIES
    );
    print_ticks(fp32_display);
    std::printf(" wide_sequence_median_us=");
    print_ticks(wide_display);
    std::printf(" packed_sequence_median_us=");
    print_ticks(packed_display);
    std::printf(" fp32_per_launch_median_us=");
    print_ticks(rounded_divide(
        summary.fp32_sequence_twice_ticks,
        2 * ROTATION_COPIES
    ));
    std::printf(" wide_per_launch_median_us=");
    print_ticks(rounded_divide(
        summary.wide_sequence_twice_ticks,
        2 * ROTATION_COPIES
    ));
    std::printf(" packed_per_launch_median_us=");
    print_ticks(rounded_divide(
        summary.packed_sequence_twice_ticks,
        2 * ROTATION_COPIES
    ));
    std::printf(
        " wide_over_packed=%.6f fp32_over_packed=%.6f\n",
        static_cast<double>(summary.wide_sequence_twice_ticks)
            / static_cast<double>(
                summary.packed_sequence_twice_ticks
            ),
        static_cast<double>(summary.fp32_sequence_twice_ticks)
            / static_cast<double>(
                summary.packed_sequence_twice_ticks
            )
    );
    ++evidence->summaries;
    return summary;
}

void emit_rotated_provenance(
    const cudaDeviceProp &properties,
    const RunIdentity &identity,
    RotatedMode mode
) {
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
        "PROVENANCE schema=p13_rotated_v1 mode=%s "
        "gpu_name_hex=%s device_uuid=%s sm_count=%d "
        "compute=%d.%d l2_bytes=%d warp_size=%d "
        "nvcc_cudart=%d runtime_version=%d driver_version=%d "
        "rotated_source_sha256=%s protocol_sha256=%s "
        "imported_bench_sha256=%s math_header_sha256=%s "
        "selftest_header_sha256=%s candidate_sha256=%s "
        "build_id=%s replicate_id=%d run_id=%s\n",
        rotated_mode_name(mode),
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
        P13_ROTATED_SOURCE_SHA256,
        P13_ROTATED_PROTOCOL_SHA256,
        P13_BENCH_SOURCE_SHA256,
        P13_MATH_HEADER_SHA256,
        P13_SELFTEST_HEADER_SHA256,
        P13_CANDIDATE_SHA256,
        P13_ROTATED_BUILD_ID,
        identity.replicate_id,
        identity.run_id.c_str()
    );
    std::printf(
        "SEMANTICS weight_storage=wide_i32_vs_packed_i16 "
        "weight_fraction_bits=12 activation=i32_f16 "
        "accumulator=signed_i64 oracle=signed_i128 "
        "epilogue=mathematical_floor measurement=rotated_sequence "
        "copies=%zu no_tensor_cores=true\n",
        ROTATION_COPIES
    );
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const RotatedMode mode = parse_rotated_mode(argc, argv);
        require_rotated_build_identity();
        const RunIdentity run_identity = require_run_identity();
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        if (properties.major != 12
            || properties.minor != 0
            || properties.warpSize != static_cast<int>(WARP_SIZE)
            || properties.l2CacheSize <= 0) {
            fail("rotated target must be sm_120 with reported L2");
        }
        emit_rotated_provenance(properties, run_identity, mode);
        run_host_controls();
        run_device_selftests();
        run_rotated_negative_control();
        run_rotated_pressure_selftest();

        if (mode == RotatedMode::SANITIZE_ONLY) {
            std::printf(
                "FINAL disposition=SANITIZER_QUALIFICATION_PASS "
                "scope=small_rotated_fixture copies=%zu "
                "retained_timing=none\n",
                ROTATION_COPIES
            );
            if (std::fflush(stdout) != 0 || std::ferror(stdout) != 0) {
                fail("rotated sanitizer stdout flush failed");
            }
            return 0;
        }

        const std::size_t l2_bytes =
            static_cast<std::size_t>(properties.l2CacheSize);
        const std::size_t pressure_bytes =
            pressure_bytes_for(properties);
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        DeviceBuffer<uint64_t> pressure(
            pressure_bytes / sizeof(uint64_t)
        );
        CUDA_CHECK(cudaMemset(
            pressure.get(), 0x35, pressure.bytes()
        ));
        RotatedPressureVerifier pressure_verifier(pressure.count());
        const PressureVerification initial_pressure =
            pressure_verifier.advance_and_verify(pressure);
        std::printf(
            "PRESSURE generation=%llu bytes=%zu l2_bytes=%zu "
            "ratio_to_l2=%.6f "
            "verified_words=%zu full_footprint=PASS "
            "checksum_fnv64=%016llx free_before_bytes=%zu "
            "total_bytes=%zu\n",
            static_cast<unsigned long long>(
                initial_pressure.generation
            ),
            pressure.bytes(),
            l2_bytes,
            static_cast<double>(pressure.bytes())
                / static_cast<double>(l2_bytes),
            initial_pressure.verified_words,
            static_cast<unsigned long long>(
                initial_pressure.checksum
            ),
            free_bytes,
            total_bytes
        );

        RotatedEvidenceCounts evidence;
        FloatingVerdict floating;
        std::vector<RotatedSummary> summaries;
        for (const Shape &shape : ROTATED_SHAPES) {
            FloatingVerdict shape_floating;
            Fixture fixture = make_fixture(shape);
            require_integer_poison_distinct(
                fixture.expected_i32, INT_POISON_A
            );
            require_integer_poison_distinct(
                fixture.expected_i32, INT_POISON_B
            );
            RotatedDeviceFixture device(fixture);
            require_rotated_working_set(device, l2_bytes);
            std::printf(
                "BOUNDARY shape=%s copies=%zu l2_bytes=%zu "
                "one_wide_bytes=%zu one_packed_bytes=%zu "
                "one_fp32_bytes=%zu rotated_wide_bytes=%zu "
                "rotated_packed_bytes=%zu rotated_fp32_bytes=%zu "
                "wide_ratio_to_l2=%.6f packed_ratio_to_l2=%.6f "
                "fp32_ratio_to_l2=%.6f wide_h2d_us=%.3f "
                "packed_h2d_us=%.3f fp32_h2d_us=%.3f "
                "activation_conversion=none\n",
                shape.id,
                ROTATION_COPIES,
                l2_bytes,
                fixture.weights_wide.size() * sizeof(int32_t),
                fixture.weights_packed.size() * sizeof(int16_t),
                fixture.weights_fp32.size() * sizeof(float),
                device.weights_wide.bytes(),
                device.weights_packed.bytes(),
                device.weights_fp32.bytes(),
                static_cast<double>(device.weights_wide.bytes())
                    / static_cast<double>(l2_bytes),
                static_cast<double>(device.weights_packed.bytes())
                    / static_cast<double>(l2_bytes),
                static_cast<double>(device.weights_fp32.bytes())
                    / static_cast<double>(l2_bytes),
                device.wide_h2d_us,
                device.packed_h2d_us,
                device.fp32_h2d_us
            );
            ++evidence.boundaries;
            std::printf(
                "PREFLIGHT shape=%s pack_identity=PASS "
                "signed128_oracle=PASS overflow_admission=PASS "
                "wide_output=PASS packed_output=PASS fp32_bound=PASS "
                "maximum_absolute_product_sum=%llu "
                "maximum_absolute_exact_sum=%llu\n",
                shape.id,
                static_cast<unsigned long long>(
                    fixture.maximum_absolute_product_sum
                ),
                static_cast<unsigned long long>(
                    fixture.maximum_absolute_exact_sum
                )
            );
            ++evidence.preflights;

            if (mode == RotatedMode::QUALIFY_ONLY) {
                for (const Lane lane : {
                        Lane::FP32, Lane::WIDE, Lane::PACKED
                    }) {
                    run_rotated_untimed(
                        lane,
                        fixture,
                        device,
                        &pressure,
                        0,
                        &shape_floating
                    );
                }
                std::printf(
                    "QUALIFICATION shape=%s copies=%zu "
                    "wide_outputs=PASS packed_outputs=PASS "
                    "fp32_outputs=PASS retained_timing=none\n",
                    shape.id,
                    ROTATION_COPIES
                );
            } else {
                summaries.push_back(run_rotated_schedule(
                    fixture,
                    device,
                    pressure,
                    pressure_verifier,
                    &shape_floating,
                    &evidence
                ));
                std::printf(
                    "POST shape=%s timed_output_slices=%zu "
                    "timed_outputs=PASS fp32_calls=%zu "
                    "fp32_elements=%zu fp32_max_error=%.12Le "
                    "fp32_max_bound=%.12Le\n",
                    shape.id,
                    static_cast<std::size_t>(12)
                        * ROTATION_COPIES * 3,
                    shape_floating.calls,
                    shape_floating.elements,
                    shape_floating.maximum_error,
                    shape_floating.maximum_bound
                );
                ++evidence.posts;
            }
            const std::size_t expected_shape_fp32_calls =
                mode == RotatedMode::QUALIFY_ONLY
                ? ROTATION_COPIES
                : 12 * ROTATION_COPIES;
            if (!shape_floating.passed()
                || shape_floating.calls
                    != expected_shape_fp32_calls) {
                fail("shape-local rotated FP32 aggregate invalid");
            }
            floating.merge(shape_floating);
            ++evidence.shapes;
        }

        if (mode == RotatedMode::QUALIFY_ONLY) {
            if (evidence.boundaries != 2
                || evidence.preflights != 2
                || evidence.shapes != 2
                || evidence.samples != 0
                || evidence.summaries != 0
                || evidence.posts != 0
                || !floating.passed()
                || floating.calls != 2 * ROTATION_COPIES) {
                fail("incomplete rotated qualification evidence");
            }
            std::printf(
                "FINAL disposition=QUALIFICATION_PASS "
                "scope=production_rotated_sequences shapes=2 "
                "copies=%zu retained_timing=none\n",
                ROTATION_COPIES
            );
        } else {
            if (evidence.boundaries != 2
                || evidence.preflights != 2
                || evidence.samples != 72
                || evidence.summaries != 2
                || evidence.posts != 2
                || evidence.shapes != 2
                || summaries.size() != 2
                || !floating.passed()
                || floating.calls
                    != 2 * 12 * ROTATION_COPIES) {
                fail("incomplete retained rotated evidence");
            }
            std::printf(
                "DIMENSIONS representation_identity=PASS "
                "integer_exactness=PASS floating_bound=PASS "
                "timed_output_integrity=PASS "
                "measurement_unit=amortized_rotated_sequence "
                "automatic_promotion=DISABLED\n"
            );
            std::printf(
                "FINAL disposition=MEASURED_NEEDS_EXTERNAL_VALIDATION "
                "scope=synthetic_rotated_m1_projection_gemv "
                "classification=representation_plus_execution "
                "shapes=2 copies=%zu samples=%zu summaries=%zu "
                "nonclaims=no_single_launch_no_layer_no_tokens_"
                "no_runtime_no_real_weights_no_tensor_cores\n",
                ROTATION_COPIES,
                evidence.samples,
                evidence.summaries
            );
        }
        if (std::fflush(stdout) != 0 || std::ferror(stdout) != 0) {
            fail("rotated stdout flush failed");
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
