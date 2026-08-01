# Full uniform-F12 MicroGPT on both Pico 2 ISA modes

**Disposition: PASS — the direct-F12 signal transfers to both execution
modes of the same RP2350.** Both modes reproduced the shared rounded-wide
F12 machine result. The no-FPU Hazard3 result independently shows that this
benefit does not require floating-point hardware.

| RP2350 mode | Wide median | Packed median | Packed time reduction | Wide → packed linked flash | Static RAM |
|---|---:|---:|---:|---:|---:|
| Cortex-M33 ARM | 3,071,021 µs | 1,413,663.5 µs | **53.968%** (2.172× throughput) | 186,840 → 101,208 B (−45.832%) | 19,608 B → unchanged |
| Hazard3 rv32imac, no FPU | 3,747,113 µs | 2,070,942.5 µs | **44.732%** (1.809× throughput) | 206,476 → 120,844 B (−41.473%) | 29,748 B → unchanged |

Every warmup and all eight calls in each `WPPWPWWP` capture matched the same
20 samples, 122 inference steps, sample hash, and canonical raw pre-temperature
logit hash as the XIAO RP2040 run. In both modes, the preselected `+1`
BOS-embedding code
left the names unchanged but produced the independently pinned different
raw-logit hash.

The timing ranges were small: ARM wide 3,070,946–3,071,361 µs and packed
1,413,527–1,414,311 µs; RISC-V wide 3,746,914–3,747,379 µs and packed
2,070,795–2,070,999 µs. Loading/reset remained outside timing; raw-logit
observation and sample hashing remained inside both lanes equally.

All six lane builds use the pinned arduino-pico platform and framework. The
RISC-V ELF declares rv32 I/M/A/C plus Z extensions with no F/D extension; its
packed `linear_fwd` uses signed-halfword `lh`. The ARM packed path uses
`ldrsh`. The ARM target has an FPU, but the model path remains the same
portable integer core and does not use floating-point model arithmetic.

The 85,632-byte absolute flash saving is identical in both modes. Different
framework/toolchain code sizes make its percentage 45.832% on ARM and 41.473%
on RISC-V. ARM-versus-RISC-V absolute timing is descriptive only: they are
different compiler/ISA targets, despite sharing the physical chip.

This is still one deterministic tiny model and one board. It does not prove a
floating-point comparison, training or quality generalization, a board matrix,
or TinyLlama transfer. The speed result combines narrower XIP/cache traffic,
signed-int16 weight loads, and execution effects; it is not isolated arithmetic
latency.

The public tree retains the shared model, firmware source, and pinned
PlatformIO profiles. Raw captures and machine-specific build products are not
part of the release.
