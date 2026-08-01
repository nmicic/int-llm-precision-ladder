# GPU results

The GPU work answers three different questions. Keeping them separate avoids
turning a storage result into an arithmetic or full-runtime claim.

## 1. Exact Q16.48 against native BF16

Five real TinyLlama matrices were tested with an exact unpacked Q16.48 integer
kernel and native BF16/cuBLAS. The integer lane passed its conversion and
signed-128 correctness checks, but the substantial shapes were 2.44--3.68x
slower than BF16. The valid result is `STOP/PARK`, not an integer-GPU win.

Details: [`GPU_REAL_WEIGHT_RESULTS.md`](GPU_REAL_WEIGHT_RESULTS.md).

## 2. Matched four-byte CUDA-core baseline

The inherited P13/P13B controls compare simple FP32 and INT32 Q16.16 kernels
with the same four-byte element width and no tensor cores. On the RTX 5090
rebaseline, the lanes remained at approximately 0.98--1.01x parity. This is
the decisive control for whether integer arithmetic alone bought speed: it
did not.

Details: [`P13_REBASELINE_RESULTS.md`](P13_REBASELINE_RESULTS.md).

## 3. Narrower resident integer storage

A signed-int16 resident-weight candidate preserved the checked integer outputs
and reduced large-shape medians, but its first single-launch protocol failed a
whole-matrix stability gate and remains parked. A separate 12-GEMV rotated
sequence over distinct resident addresses passed and reduced sequence medians
by 41.2--42.1% relative to identical codes stored as int32.

That is evidence for representation and memory traffic. It is synthetic and
does not establish a real-weight, layer, token-generation, or end-to-end
runtime advantage.

Details: [`P13_INT16_RESULTS.md`](P13_INT16_RESULTS.md) and
[`P13_ROTATED_RESULTS.md`](P13_ROTATED_RESULTS.md).

## Verdict

The exact integer path does not compete with the formats and libraries the GPU
is optimized for. Narrow storage can help bandwidth-sensitive synthetic
kernels, but no retained result turns that signal into a TinyLlama GPU runtime
win. GPU work is therefore parked; the CUDA sources remain under `spikes/` as
reference implementations and negative evidence.
