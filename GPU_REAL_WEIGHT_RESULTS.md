# Native BF16 versus exact Q16.48 CUDA result

Run date: 2026-07-30

Target: NVIDIA GeForce RTX 5090, compute capability 12.0, CUDA 13.3

Scope: five real TinyLlama weight matrices and two frozen
embedding-derived probe vectors; kernel time only

## Outcome

**VALID evidence; STOP/PARK the current exact, unpacked Q16.48 CUDA lane.**

The run validates the declared correctness and evidence gates. It does not
show a useful exact-integer GPU advantage. On the three substantial matrix
shapes, the fixed Q16.48 kernel took 2.44--3.68x as long as native
BF16/cuBLAS.

This is not a layer, token, full-runtime, or best-possible-integer-kernel
result.

## Relationship to the earlier CUDA baseline

This run does not supersede the public P13/P13B experiment in
`int-llm/gpu/`. P13 is the controlled arithmetic comparison:

- FP32 and INT32 Q16.16 both use 4-byte inputs;
- both use custom CUDA-core-only kernels with the same structure and access
  pattern;
- neither uses cuBLAS, WMMA or tensor cores;
- the recorded M=1 ratios are 0.98--1.00x, i.e. parity for the tested simple
  kernels.

P13B extends those matched simple kernels through M=64 and finds the same
memory-latency-limited parity. Those kernels are a baseline, not an optimized
ceiling.

The present BF16-versus-Q16.48 run instead verifies real-artifact
correspondence, exact integer arithmetic and the expected hardware asymmetry.
It confirms an already-known dead end: 8-byte unpacked Q16.48 weights on CUDA
cores do not compete with 2-byte BF16 on its optimized hardware path.

If GPU exploration resumes, the useful performance question is whether a new
integer representation or execution route can lower the inherited P13 INT32
times while retaining an unchanged FP32 control and an explicit correctness
oracle. If storage width changes, any gain is a
representation-plus-execution result, not evidence that integer arithmetic
itself became faster.

## What was compared

- `FLOATING-REFERENCE`: native TinyLlama BF16 safetensors weights and BF16
  probes, cuBLAS with FP32 accumulation/output; tensor-core eligible.
- `INTEGER-CORE`: corresponding Q16.48 MGW int64 weights and probes, exact
  signed-128 accumulation and one final floor shift; no integer tensor-core
  path.
- five shapes: attention K, attention Q, MLP gate, MLP down, and the embedding
  matrix;
- two fixed probe slices for correctness; probe 0 for timing;
- 16 fully verified single-launch samples per lane, tensor, and regime.

The Q16.48 implementation was one fixed, untuned kernel. The comparison is
intentionally asymmetric because that is the hardware reality of these
representations.

## Provenance

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| TinyLlama BF16 safetensors | 2,200,119,864 | `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933` |
| Q16.48 MGW | 8,800,406,496 | `7e8218d7f79a784f9d1868140fb16c3b9f5fbc45c19fb5c807ddcba5b41e32a8` |
| CUDA source | 62,237 | `858b0e8d37b3b4031ae3467f48c61e47aa303e0b31c5acabca8fe5c0da3746f4` |
| v2 build manifest | 773 | `221a080acf0250c832cb95117aa0737f84de48fa4cc675a55f440609559024bf` |
| v2 executable | 1,198,120 | `2a90b38da256d828d97c058e9692a7a740d9e60376f2a8e6aa9d61ad6ef37422` |
| frozen run log | 60,332 | `22fd6231693eda38a5e0a47ef0abd88d001e9daee1c8c3b7e3f36496a7280ed3` |
| exit record | 14 | `92fa48783974cb041ff4ac4dea12ce713dc121a8e7109e7a2f5a6e722b9e30cf` |

The reviewed build manifest binds the exact CUDA source, math header, imported
loader header, build script, nvcc 13.3.73 executable, recipe, and binary.
The runner opened the manifest, binary, and both model artifacts first, then
hashed and used those same descriptors.

## Correctness and representation

- all 93,323,264 selected BF16 weights matched the MGW int64 values after the
  pinned repository conversion;
- RW0--RW3 were exactly representable on the Q48 grid;
- RW4 had 225,280 sub-grid values out of 65,536,000 (0.34375%), all
  intentionally converted to zero;
- both frozen probe inputs had zero sub-grid values;
- every integer result matched the independent signed-128 CPU oracle exactly;
- every BF16 result stayed inside its predeclared row-wise FP32 accumulation
  bound with zero non-finite or sentinel hits;
- all 320 timed outputs were fully verified, and all post-timing
  distinct-poison checks passed.

The largest displayed BF16 error-to-bound ratio was about `3.0143e-4`, giving
roughly 3,317x headroom against the conservative bound.

Repository-conversion identity is not a lossless-BF16 claim. The floating
oracle decodes native BF16 dyadic values independently of Q16.48; the
conversion rounding census is reported separately.

## Timing

`Integer/BF16` is the ratio of medians within the same predeclared protocol
label. Lower is better.

| Tensor | Regime label | BF16 median us | Q16.48 median us | Integer/BF16 |
|---|---|---:|---:|---:|
| RW0 | repeat hot | 7.920 | 7.776 | 0.982 |
| RW0 | after L2 pressure | 10.112 | 11.856 | 1.172 |
| RW1 | repeat hot | 12.288 | 12.288 | 1.000 |
| RW1 | after L2 pressure | 18.400 | 26.640 | 1.448 |
| RW2 | repeat hot | 24.640 | 60.224 | 2.444 |
| RW2 | after L2 pressure | 25.600 | 66.032 | 2.579 |
| RW3 | repeat hot | 24.064 | 62.464 | 2.596 |
| RW3 | after L2 pressure | 26.624 | 67.584 | 2.538 |
| RW4 | repeat hot | 89.184 | 314.864 | 3.530 |
| RW4 | after L2 pressure | 93.168 | 343.312 | 3.685 |

The tiny RW0 hot result is a 0.144-us nominal integer edge amid overlapping
distributions and reverses under the separate pressure label. RW1 ties in the
hot label. Neither is retained as an integer win.

RW2/RW3 hot samples expose schedule-position/cache-residency bimodality. Raw
samples are retained, and the fixed sequence gives each lane one adjacent
same-lane pair per eight-sample block. The source explicitly marks
cross-regime comparison invalid. The same-regime direction remains unchanged;
pressure-labelled samples and RW4 are much tighter.

RW4 is the clearest large-shape result. Both weight images exceed the reported
L2, and Q16.48 is 3.53--3.68x slower. Its int64 image is exactly four times the
BF16 bytes. The integer kernel services nominal bytes somewhat faster, but not
four times faster, so the fourfold weight image is the dominant structural
constraint visible in the elapsed result.

## Limits

- one GPU, process run, timed probe, and frozen implementation;
- 16 observations per lane/regime; no confidence-interval or population
  claim;
- event time excludes loading, conversion, host/device transfer,
  verification, layer composition, and token generation;
- `repeat hot` and `after L2 pressure` are protocol labels, not proven cache
  states, and must not be compared causally;
- nominal weight GB/s is byte accounting, not a hardware-counter DRAM
  measurement;
- a better exact integer kernel is not logically ruled out, but the fourfold
  int64 weight image is structural and does not justify tuning this lane.

## Disposition and next boundary

Do not tune or rerun this exact unpacked Q16.48 kernel. A packed,
lower-precision, or fused candidate would ask a materially different question
and needs its own oracle,
representation transitions, boundary-inclusive costs, and non-claims. It
cannot inherit this result as proof of numerical identity to the original
Q16.48 model.

Such a spike should start from the P13/P13B FP32-versus-INT32 CUDA-core
baseline, not from the BF16/cuBLAS timing in this report.

Machine-local raw runs, build logs, and binaries are not part of the public
tree. The exact CUDA source used by the retained run is included under
`spikes/gpu-real-weight/` and matches the hash above.
