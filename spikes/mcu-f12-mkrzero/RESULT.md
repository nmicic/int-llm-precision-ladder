# Full uniform-F12 MicroGPT on Arduino MKR Zero

**Disposition: correctness and storage PASS; no material runtime transfer.**
The shared rounded-wide and direct-int16 F12 lanes produced the same
20 samples, 122 inference steps, and raw pre-temperature logit hash on the
no-FPU SAMD21. The packed representation saved substantial flash but changed
full-workload time by only 0.028%, which is parity for this experiment.

| Target | Wide median | Packed median | Packed time reduction | Wide → packed linked flash | Static RAM |
|---|---:|---:|---:|---:|---:|
| MKR Zero, Cortex-M0+ @ 48 MHz, no FPU | 26,556,100.5 µs | 26,548,753.0 µs | **0.027668%** (1.000277× throughput) | 135,188 → 49,556 B (**−63.343%**) | 13,220 B → unchanged |

Every warmup and all eight calls in the challenged `WPPWPWWP` capture matched
the shared sample and raw-logit pins. The preselected one-code image reproduced
its independently pinned different raw-logit hash while leaving the sampled
names unchanged. The wide calls ranged from 26,556,079 to 26,556,133 µs; the
packed calls ranged from 26,548,739 to 26,548,768 µs. The tiny difference is
stable but not useful as a runtime improvement.

All three firmware variants fit the 256 KiB device. The dual comparison image
uses 195,440 B linked flash and 13,220 B static RAM. The lane-only packed image
saves the same 85,632 absolute bytes as the RP2040/Pico 2 runs because it
embeds the same
physical model roles; its larger percentage here reflects the smaller SAMD
framework/code baseline.

The packed ELF declares Thumb-1 Armv6S-M, has no FP architecture attribute,
and `linear_fwd` uses `ldrsh` before the portable wide multiply. This is a
genuine no-FPU direct-int16 execution result, not merely a compressed file
that expands to wide weights first.

The result bounds the earlier MCU findings: compact direct-F12 storage
transfers, but runtime acceleration is not universal even across no-FPU
Armv6-M boards. The contrast with RP2040/RP2350 is consistent with substantial
platform memory/XIP, cache, compiler, or instruction-cost effects; this is a
mechanism hypothesis, not a causal isolation result.

This remains exact relative to the rounded uniform-F12 oracle, not the
original Q16.48 weights. It is not a floating-point comparison, training or
quality result, general board benchmark, or TinyLlama transfer.

The public tree retains the shared model, firmware source, and pinned
PlatformIO profile. Raw captures and machine-specific build products are not
part of the release.
