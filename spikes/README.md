# Technical experiment sources

This directory retains the compact technical subset needed to understand or
rebuild the published experiments:

- `safetensors-direct-mgwi/` — pinned direct-conversion and correctness gates;
- `mcu-f12-*` — shared uniform-F12 MicroGPT models, firmware sources, and
  PlatformIO profiles;
- `gpu-real-weight/` — real-matrix exact-Q16.48/BF16 CUDA comparison source;
- `gpu-p13-next/` and `gpu-p13-rotated/` — matched-width and narrow-resident
  CUDA reference kernels.

Machine-specific binaries, raw upload/benchmark logs, device identifiers,
review records, and orchestration files are not part of the publication tree.
The retained result boundary is documented in the root result files.
