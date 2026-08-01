# P13 signed-int16 resident-weight CUDA result

Capture time: 2026-07-30T23:27Z
Target: NVIDIA GeForce RTX 5090, compute capability 12.0
Toolchain: CUDA 13.3, nvcc 13.3.73, driver 610.43.02
Scope: synthetic M=1 projection GEMV; CUDA events measure one kernel launch

## Outcome

**VALID synthetic evidence; predeclared criteria FAIL; STOP/PARK this
candidate.**

The candidate produced a strong, repeatable representation-level signal in
the two pressure-labelled decision shapes: storing the same signed weight
codes in int16 rather than int32 reduced median kernel time by 35.2--38.9%
across all three fresh processes. All 72 predeclared packed/wide pairs for
those shapes met the 5% improvement rule.

The candidate nevertheless does not survive its complete frozen gate. Seven
of the 576 retained packed/wide samples fell outside the required
lane-median +/-20% stability envelope. All seven were in the separate
`repeat_hot` regime. The threshold is not changed after seeing the result, and
the candidate is not promoted to real weights or a full runtime.

This is evidence that halving resident weight width can materially reduce
these synthetic memory-pressure GEMV times. It is not evidence that integer
arithmetic itself is faster.

## Compared lanes

All three lanes used one warp per output row and the same logical matrix and
launch geometry:

- `WIDE_INT32_CONTROL`: canonical signed weight codes stored in int32;
- `PACKED_INT16_CANDIDATE`: the identical codes stored in int16 and
  sign-extended on load;
- `FP32_CONTROL`: four-byte FP32 CUDA-core control.

The two integer lanes used the same int32 activations, signed-int64
accumulation, and mathematical-floor epilogue. The host oracle accumulated in
signed 128 bits. No lane used cuBLAS, WMMA, or tensor cores.

The int16 result is therefore classified
`representation_plus_execution`. Geometry improvements shared by both integer
lanes are not credited to packed storage.

## Decision-shape timing

Each row is the median of 12 verified single-launch observations after a
verified 256 MiB pressure traversal. Reduction is
`1 - packed / wide`.

| Shape | Process | Wide int32 us | Packed int16 us | Reduction |
|---|---:|---:|---:|---:|
| gate/up | 0 | 34.784 | 22.528 | 35.23% |
| gate/up | 1 | 32.768 | 20.480 | 37.50% |
| gate/up | 2 | 34.784 | 22.496 | 35.33% |
| down | 0 | 36.832 | 22.496 | 38.92% |
| down | 1 | 35.824 | 22.496 | 37.20% |
| down | 2 | 36.832 | 22.512 | 38.88% |

Every one of the 12 paired observations per shape and process favored packed
storage by at least 5%. Across the 72 pairs, individual reductions ranged
from 29.35% to 41.21%; every packed/wide pair also remained inside the
predeclared storage-plausibility interval.

## Secondary observations

The other median results support a traffic-versus-launch-overhead reading but
are not promotion evidence:

- q/o projection was 24.71--25.20% faster with packed storage in both
  protocol labels;
- K/V projection was effectively tied, consistent with its small working set;
- gate/up `repeat_hot` medians ranged from 7.13% to 17.06% faster;
- down `repeat_hot` was 16.71% faster in every process.

These are within-regime comparisons only. `repeat_hot` and
`after_l2_pressure` are protocol labels and are not compared causally.

## Why the frozen gate failed

The validator required every retained packed and wide sample, across all four
shapes and both protocol labels, to lie within 20% of its own lane median.
The seven failures were:

- process 1: one K/V packed spike;
- process 1: two gate/up wide low samples and one gate/up packed low sample;
- process 2: one q/o wide spike and two q/o packed spikes.

Three low samples missed the boundary only narrowly. Four were clear timing
excursions, including one q/o packed observation at 15.360 us against a
6.144-us lane median. All output checks still passed. The scientific result
is therefore stable pressure-shape medians plus an unsuccessful whole-matrix
single-launch stability gate—not a correctness failure and not a retained
performance pass.

## Correctness and path evidence

- lossless int16 pack/decode identity passed;
- signed-128 expected values, int64 safety bounds, signed-floor cases, odd
  tails, extrema, cancellation, poison, skipped-write, partial-write, and
  zero-extension negative controls passed;
- all 864 retained timed outputs across the three processes were completely
  verified;
- all FP32 outputs were finite and inside their predeclared bounds;
- a separate non-timed qualification run emitted no retained `SAMPLE`
  records and completed under Compute Sanitizer with zero errors;
- the locked SASS inspection found the intended signed-int16 and int32 load
  paths, identical normalized integer opcode sequences, 32 registers and no
  stack/local/shared allocation for both integer kernels;
- the complete disassembly contained no matrix/tensor opcode.

Resident weight bytes and diagnostic H2D bytes were exactly halved. Host
packing was a disclosed one-time cost, and there was no per-invocation
activation conversion. Packing and transfer time were not included in kernel
event time.

## Evidence identity

The trusted runner required `Exclusive_Process`, observed no other compute
process before or after any replica, pinned the physical GPU UUID, used a
minimal fixed environment, and restored the server's compute mode to
`Default` after capture.

| Artifact | SHA-256 |
|---|---|
| benchmark source | `6b6285fa1ca0c8d3128598bcb6c2632d4d2026a3fa4590fef0c5b76cf3fb2001` |
| candidate specification | `87b5ea47b2d1ece4746e3f21dd598363b1d45dae2f7c3172906b52201b4f21d3` |
| executable | `2e05e0e705c7f40efc22020d2e1cc4ad75d9b80676003d1cfd753a6b5bd77fd7` |
| SASS | `3493e9156c8774e41af20869ef761bc819c5c7449e43e1aa627e106b48080666` |
| run manifest | `b121efc9126c5958aa4ad1df7e1b0b055539a27c8874424e88c3e214e683ddf3` |
| environment record | `846225dc628d91613830f21d5bc62f831bd5f4682d0161fc828de98d8844340b` |
| validator result | `de786aeec3f4d3a4ad40636530df7933a6fb859ad1a4031e703b139a2985b6bc` |
| process 0 log | `3689a97ec209a66aa36ef2d2155540ff2a7f9645bbf81142aaba8bfd1711f2e0` |
| process 1 log | `f6feae1dfe0873290500a9ce081d986c1e69e33d91eda05e2d84ce4648f5d207` |
| process 2 log | `0caccc541bde4290a0d35f6bae7a1d4fe30ac7a6c47a429f9dbdab923bff0d01` |

Machine-local build artifacts, SASS, raw logs, and environment snapshots are
not part of the public tree. The exact benchmark source and host arithmetic
tests are retained under `spikes/gpu-p13-next/`.

## Limits and disposition

- synthetic deterministic fixtures, not serialized or real model tensors;
- one GPU and one bounded three-process capture;
- single-launch kernel event time, not a layer, token, or runtime;
- no model behavior or original-Q16.48 numerical-equivalence claim;
- FP32 is contextual control, not a tensor-core/vendor-library ceiling;
- one-time packing and transfer are disclosed diagnostics, not amortization
  claims.

Do not tune the threshold, selectively discard the seven samples, rerun this
exact protocol until it passes, or open the real TinyLlama fixture on the
strength of this result. Any later CUDA experiment must retain this
`STOP/PARK` outcome unchanged.
