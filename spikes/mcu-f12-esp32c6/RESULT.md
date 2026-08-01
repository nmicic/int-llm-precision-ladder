# Full uniform-F12 MicroGPT on ESP32-C6

**Disposition: correctness, storage, and within-target runtime PASS.** The
shared rounded-wide and direct-int16 F12 lanes produced the same 20
samples, 122 inference steps, and raw pre-temperature logit hash on the
no-FPU ESP32-C6. Packed storage materially reduced both flash and complete
inference time.

| Target | Wide median | Packed median | Packed time reduction | Wide → packed linked flash | Static RAM |
|---|---:|---:|---:|---:|---:|
| ESP32-C6, RV32IMAC @ 160 MHz, no FPU | 2,039,419.0 µs | 1,637,116.0 µs | **19.726353%** (1.245739× throughput) | 363,225 → 277,593 B (**−23.575%**) | 23,644 B → unchanged |

Both warmups and all eight calls in the challenged `WPPWPWWP` capture
matched the shared sample and raw-logit pins. The preselected one-code image
reproduced its independently pinned different raw-logit hash while leaving
the sampled names unchanged. Wide calls ranged from 2,039,406 to 2,039,434 µs;
packed calls ranged from 1,636,784 to 1,637,124 µs.

All three firmware variants fit the target. The dual comparison image uses
423,831 B linked flash and 23,644 B static RAM. The lane-only packed image
saves the same 85,632 absolute bytes as the earlier transfers because it
embeds the same physical model roles.

The exact MicroGPT model object is tagged RV32IMAC with no F or D extension,
and packed `linear_fwd` contains signed-halfword `lh` loads into the
portable wide multiply path. The fully linked pioarduino ELF inherits an F/D
architecture attribute from prebuilt framework/toolchain inputs, so that
merged tag is not used as the no-FPU proof; a complete linked-firmware
disassembly scan found zero floating-point instructions. The ESP32-C6 itself
has no FPU.

This result strengthens the useful but narrow MCU signal: direct signed-int16
F12 storage can preserve the rounded machine result and improve full-workload
time, but the MKR Zero parity result proves that no-FPU status alone does not
predict the speedup. Flash mapping/cache behavior, compiler output, and core
implementation remain plausible moderators; this experiment does not isolate
them.

This remains exact relative to the rounded uniform-F12 candidate, not the
original Q16.48 weights. It is not a floating-point comparison, arithmetic-
only attribution, training or quality result, board-family benchmark, or
TinyLlama transfer.

The public tree retains the shared model, firmware source, and pinned
PlatformIO profile. Raw captures, unit identifiers, and machine-specific build
products are not part of the release.
