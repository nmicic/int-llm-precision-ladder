# int-llm precision ladder

This repository is a follow-up to
[`int-llm`](https://github.com/nmicic/int-llm). It asks a deliberately narrow
question: how far can fixed-point model weights be rounded and stored more
compactly while preserving the behavior of the Q16.48 integer oracle on
explicit regression gates?

The main result is a mixed-F11/F12 TinyLlama-1.1B candidate whose weights fit
signed int16. It matched the Q16.48 oracle on all **632 actually evaluated
greedy decisions across 32 prompts** (640 decisions were requested; five
prompts reached matching EOS early). A direct safetensors-to-MGWI conversion
reproduced all **1,100,048,384** retained integer codes and every output byte
of the reference packed file.

This is an experimental reference, not a production quantizer or inference
runtime.

## What F12 means

`F12` means 12 fractional bits after deterministic fixed-point rounding. It
does **not** mean a 12-bit container. In the compact path each F11 or F12 code
occupies one little-endian signed 16-bit integer.

Only weights are narrowed:

- model weights: mixed F11/F12 signed-int16 codes;
- activations, KV cache, nonlinear functions, and other state: Q16.48;
- matrix accumulation: signed 128-bit intermediate;
- comparison oracle: the original Q16.48 integer runtime.

The rounded model is not numerically identical to the original oracle. The
exact claim is limited to the checked greedy decisions. Raw-logit identity is
claimed only between the rounded-wide and packed forms of the same candidate.

## Results at a glance

| Experiment | Result | Important boundary |
| --- | --- | --- |
| MicroGPT uniform F12 | 20/20 samples and 122/122 greedy decisions; packed and rounded-wide raw logits identical | Small screening model |
| TinyLlama mixed F11/F12 | 32 prompts, 632/632 evaluated greedy decisions | Short deterministic gate, not general quality parity |
| Direct TinyLlama conversion | 2,200,116,192-byte MGWI, byte-identical to the reference packed file | Pinned single-file TinyLlama checkpoint and plan |
| AMD packed vs rounded-wide | 23.209% lower generation time; 73.303% lower peak RSS | Same integer model, one AMD host |
| AMD packed vs FP32 runtime | FP32 was 3.168899x faster; packed used about 69.1% less peak RSS | Different runtime stacks; not arithmetic-only |

The compact TinyLlama file is 2,200,116,192 bytes, compared with
8,800,406,496 bytes for Q16.48 MGW and 2,200,119,864 bytes for the source
safetensors file. The representation removes the fourfold storage expansion
introduced by the conservative Q16.48 oracle.

See [`TINYLLAMA_RESULTS.md`](TINYLLAMA_RESULTS.md) for the full model result and
[`RESULTS.md`](RESULTS.md) for the small-model precision sweep.

## MCU transfer

The complete uniform-F12 MicroGPT workload was also run from flash on four
physical boards in five ISA configurations. Every packed lane reproduced the
same 20 samples, 122 inference steps, and canonical raw-logit hash as its
rounded-wide F12 control.

| Target | FPU | Linked-flash reduction | Full-run time reduction |
| --- | --- | ---: | ---: |
| Seeed XIAO RP2040, Cortex-M0+ | No | 45.29% | 17.335% |
| Raspberry Pi Pico 2, Cortex-M33 | Present, not used by the model path | 45.832% | 53.968% |
| Raspberry Pi Pico 2, Hazard3 RV32IMAC | No | 41.473% | 44.732% |
| Arduino MKR Zero, Cortex-M0+ | No | 63.343% | 0.027668% (parity) |
| ESP32-C6, RV32IMAC | No | 23.575% | 19.726353% |

These are within-target packed-versus-rounded-wide comparisons. They are not
floating-point comparisons, and the MKR Zero result shows that narrower
storage does not guarantee a speedup even without an FPU. Firmware sources,
generated models, and pinned PlatformIO profiles are under `spikes/mcu-*`;
the measurements are summarized in [`MCU_RESULTS.md`](MCU_RESULTS.md).

## GPU result: still parked

The GPU evidence remains negative or narrowly mixed:

- exact unpacked Q16.48 integer kernels were **2.44--3.68x slower** than native
  BF16/cuBLAS on substantial real TinyLlama matrix shapes;
- matched four-byte FP32 and INT32 Q16.16 CUDA-core controls remained at
  roughly **0.98--1.01x parity**, so integer arithmetic alone showed no useful
  advantage;
- storing identical synthetic integer codes as resident int16 instead of
  int32 reduced a rotated 12-GEMV sequence by **41.2--42.1%**, but this is a
  representation/bandwidth result, not a model, layer, token, or runtime win.

There is no GPU game-changer in these results. The CUDA code is retained as
useful positive and negative reference material; see
[`GPU_RESULTS.md`](GPU_RESULTS.md).

## Run the local validation

The self-contained, non-network entry point is:

```sh
sh scripts/validate.sh local
```

It builds the native tools, runs the unit tests, checks the committed
MicroGPT model, validates the portable/native F12 host fixture, and runs the
host-side GPU arithmetic tests. It does not use SSH, flash hardware, download
TinyLlama, or rerun CUDA measurements.

Useful individual commands:

```sh
make test
make inspect
make sweep
make sensitivity
make mixed
make llama-mgwi
python3 spikes/mcu-f12-microgpt/prepare.py
```

Generated small-model result tables are committed under `results/`. Large
TinyLlama weights are intentionally not stored in this repository.

## Direct TinyLlama conversion

The pinned direct converter is `tools/safetensors_to_mgwi.py`. The retained
gate can be rerun when the exact external inputs are available:

```sh
python3 spikes/safetensors-direct-mgwi/run_gate.py \
  --model-dir /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --retained-mgwi /path/to/reference.mgwi \
  --rounded-mgw /path/to/rounded.mgw \
  --output-dir /path/to/new-result
```

The behavioral gate is in
`spikes/safetensors-direct-mgwi/run_correctness_gate.py`. Both runners pin the
expected sizes and SHA-256 identities and fail closed on a mismatch.

MGWI is currently a trusted-input experimental format. The loader and tools
are not offered as a hostile-file security boundary.

## Repository layout

- `src/` — MicroGPT and TinyLlama integer runtimes;
- `tools/` — precision sweeps, MGW rounding, MGWI packing, and direct
  safetensors conversion;
- `tests/` — self-contained numerical and format tests;
- `spikes/safetensors-direct-mgwi/` — retained full-conversion and correctness
  gates;
- `spikes/mcu-*` — compact MicroGPT firmware sources and build profiles;
- `spikes/gpu-*` — CUDA reference kernels and host arithmetic tests;
- `results/` — compact machine-readable small-model results;
- `scripts/validate.sh` — publication-tree acceptance command.

## Limits

- The TinyLlama claim covers short greedy regression gates, not arbitrary
  prompts, sampling, or chat-quality equivalence.
- The 23.209% time improvement compares packed and rounded-wide integer paths
  on one AMD host. It is not a comparison with FP32.
- The practical FP32 control was substantially faster than the current packed
  C runtime. The compact representation's strongest x86 result is memory and
  file size, not speed.
- The MCU results compare two integer representations of one tiny model. No
  MCU floating-point comparison is claimed.
- The CUDA results do not establish an integer inference advantage.
- The Q16.48 path remains the behavioral oracle; lower precision is a checked
  candidate, not a replacement truth source.

See [`PROVENANCE.md`](PROVENANCE.md) and [`VALIDATION.md`](VALIDATION.md) for
source identities and the exact validation boundary.

## References

- [`int-llm` source repository](https://github.com/nmicic/int-llm)
- [`int-llm-coordinate-permutation`: reversible model-layout experiments across vocabulary, neurons, attention heads, and hidden coordinates](https://github.com/nmicic/int-llm-coordinate-permutation)
- [Published `int-llm` model and `model.mgw` artifact](https://huggingface.co/nmicic/int-llm)
- [Original article: *int-llm: a pure-integer LLM experiment in C*](https://huggingface.co/blog/nmicic/int-llm)
