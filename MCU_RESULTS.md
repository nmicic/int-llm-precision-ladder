# MCU uniform-F12 results

Run date: 2026-07-31

The committed 115,576-byte MicroGPT model was rounded uniformly to F12 and
represented in two execution forms:

- rounded-wide: F12 values retained in the Q16.48/int64 MGW layout;
- packed: the same F12 codes stored directly as signed int16 in MGWI.

The packed model is 29,944 bytes, a reduction of 85,632 bytes (74.09%) from
the rounded-wide model container. Activations and arithmetic results remain
Q16.48. Every physical packed run reproduced the rounded-wide control's 20
samples, 122 inference steps, sample hash, and canonical raw pre-temperature
logit hash. A one-code corrupted model kept the same displayed samples but
changed the raw-logit hash and was rejected, demonstrating that the gate was
sensitive below the sampled-output boundary.

## Physical results

| Target | ISA | FPU boundary | Wide flash | Packed flash | Flash reduction | Wide median | Packed median | Time reduction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seeed XIAO RP2040 | Cortex-M0+ | No FPU | 189,092 B | 103,460 B | 45.29% | 6,714,832 us | 5,550,795 us | 17.335% |
| Raspberry Pi Pico 2 ARM mode | Cortex-M33 | Model path used portable integer core | 186,840 B | 101,208 B | 45.832% | 3,071,021 us | 1,413,663.5 us | 53.968% |
| Raspberry Pi Pico 2 RISC-V mode | Hazard3 RV32IMAC | No FPU | 206,476 B | 120,844 B | 41.473% | 3,747,113 us | 2,070,942.5 us | 44.732% |
| Arduino MKR Zero | Cortex-M0+ | No FPU | 135,188 B | 49,556 B | 63.343% | 26,556,100.5 us | 26,548,753.0 us | 0.027668% |
| ESP32-C6 | RV32IMAC | No FPU | 363,225 B | 277,593 B | 23.575% | 2,039,419.0 us | 1,637,116.0 us | 19.726353% |

Static RAM was unchanged within each target. Loading and reset were outside
the timed boundary; raw-logit observation and sample hashing were inside both
lanes. Each physical comparison used the same alternating `WPPWPWWP` schedule
after warmup.

The packed paths used signed-halfword loads (`ldrsh` on ARM and `lh` on
RISC-V). The models were executed directly from the narrow representation;
they were not expanded into an int64 weight copy in RAM first.

## Interpretation

The portable conclusion is storage, exactness relative to the rounded F12
oracle, and feasibility on no-FPU devices. Runtime improvement is
platform-dependent: three targets showed a material reduction, while the MKR
Zero was effectively tied. Cache/XIP behavior, compiler output, memory layout,
and core instruction costs are all part of these measurements.

These results do not compare integer execution with floating point and do not
transfer the TinyLlama quality claim to an MCU.
They cover one deterministic tiny model and the listed physical targets.

## Rebuild

Regenerate and verify the shared model fixture locally:

```sh
python3 spikes/mcu-f12-microgpt/prepare.py
```

Then build a selected PlatformIO profile, for example:

```sh
pio run -d spikes/mcu-f12-microgpt/harness -e xiao_rp2040_dual
pio run -d spikes/mcu-f12-pico2-dual-isa/harness -e pico2_riscv_dual
pio run -d spikes/mcu-f12-mkrzero/harness -e mkrzero_dual
pio run -d spikes/mcu-f12-esp32c6/harness -e esp32c6_dual
```

The public tree retains source and build profiles, not unit identifiers,
upload logs, machine-local build maps, or prebuilt firmware images.
