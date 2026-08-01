# First results

Run date: 2026-07-30

The committed 115,576-byte MicroGPT model provides a useful fast oracle. A
complete 49-point weight sweep from F48 through F0, including inference and
trace comparison for every candidate, finishes in under one second on the
Mac CPU. No GPU path is involved.

## Uniform weight ladder

All 14,272 model weights were rounded while inference arithmetic, activations,
accumulators, tokenizer data and PRNG state remained unchanged at Q16.48.

| Fraction bits | Samples | Sampled tokens | Top-1 tokens | Observation |
| ---: | ---: | ---: | ---: | --- |
| 48 | 20/20 | 122/122 | 122/122 | Byte-identical model and trace |
| 24 | 20/20 | 122/122 | 122/122 | Raw logits differ from the oracle |
| 16 | 20/20 | 122/122 | 122/122 | Output behavior still exact |
| 12 | 20/20 | 122/122 | 122/122 | Lowest tested uniform grid with exact top-1 behavior |
| 11 | 20/20 | 122/122 | 121/122 | First top-1 change |
| 10 | 20/20 | 122/122 | 121/122 | Same 20 stochastic samples, but not greedy-exact |
| 9 | 18/20 | 113/122 | 115/122 | First sampled-output drift |
| 6 | 2/20 | 27/115 | 55/115 | Broad behavioral collapse |
| 0 | 0/20 | 20/114 | 44/114 | Integer-grid endpoint |

The first raw-logit hash change occurs immediately at F47. That is expected:
the integer-trained model uses low bits. The useful result is the wide gap
between internal numerical divergence and externally visible behavior.

At F11 and F10 the only top-1 mismatch is sample 15, position 5. The sampled
token remains identical, so the final name still matches. This explains why
20/20 stochastic samples is weaker evidence than the 122/122 top-1 result.

Sample-match counts below the boundary are not monotonic. The sampler uses a
fixed PRNG stream, and a coarser candidate can cross a cumulative-probability
threshold in either direction. For example, F7 happens to reproduce 19/20
whole samples while F8 reproduces 15/20; that does not make F7 numerically
better.

## Tensor sensitivity near the boundary

Seven logical tensor groups were tested at F11, F10 and F9 in two ways:

- `only`: round that group and leave every other tensor at F48;
- `rescue`: round every other group and keep that group at F48.

At F9, no single rounded group changes a sampled name. The full F9 model does,
which makes the failure an interaction between perturbations rather than a
single isolated bad tensor.

The language-model head is the clearest lever:

- rounding only the head at F9 changes one top-1 decision but no sampled name;
- keeping only the head at F48 while every other tensor is F9 restores all
  20 samples, all 122 sampled tokens and all 122 top-1 decisions.

The rescue results are nonlinear. Keeping attention Q/K/V at F48 while all
other groups are F9 is worse than uniformly rounding everything to F9. This
is evidence of error cancellation, so individual-tensor results must not be
read as an independent ranking of layer importance.

## What this signalled

The tiny oracle supported this promotion sequence:

1. treat F12 as the conservative exact-behavior point for this model;
2. probe mixed precision around F12/F11/F10/F9, especially the output head;
3. promote only a few candidates to TinyLlama, first on one greedy prompt and
   then on the complete 80-token gate.

This section is screening evidence, not a TinyLlama result. TinyLlama starts from
converted floating-point weights and has a different scale and architecture,
so its boundary may occur elsewhere. This experiment also emulates precision
inside 64-bit Q16.48 storage; by itself it does not demonstrate smaller files,
narrow kernels or a speedup.

## Mixed body/head follow-up

A second grid varied the language-model head independently from the remaining
weights:

- body F11 plus head F12 restores 122/122 top-1 agreement;
- body F9 plus any tested head setting from F10 through F48 also gives
  122/122;
- body F8 succeeds only at isolated head settings (F10, F13 and F14), while
  both more and less head precision can fail.

The last result is a useful warning. Exact output at one mixed setting may be
caused by cancellation rather than a generally better representation. Body
F9 also passes while body F10 does not, even with the head exact. Therefore
uniform F12 remains the conservative tiny-model promotion point; the exact
mixed settings are sensitivity probes, not yet compression recipes.

The streaming C converter used for TinyLlama is tested byte-for-byte against
the Python mixed-plan oracle, including F48 identity and an F9/F12 mixed
candidate.

The complete machine-readable results are in
`results/weight_ladder.{csv,json}` and
`results/tensor_sensitivity.{csv,json}` and
`results/mixed_ladder.{csv,json}`.

## Promotion outcome

The TinyLlama follow-up found a mixed F11/F12 candidate that remained exact
on 32 prompts / 632 of 632 actually evaluated greedy decisions (640 maximum
requested). Its values fit signed int16, and the
experimental direct-int16 path reproduced 3,264,000 raw int64 logits exactly
while reducing median generation time by 23.209% and peak RSS by 73.303% in
the admitted AMD packed-versus-rounded-wide comparison. See
[`TINYLLAMA_RESULTS.md`](TINYLLAMA_RESULTS.md); those results do
not retroactively make the tiny-model cancellation islands robust.

## Exact Q16.48 CUDA baseline

A bounded real-weight CUDA experiment compared native BF16/cuBLAS with
the exact unpacked Q16.48 MGW lane on five TinyLlama matrices. All converted
weights and probe inputs passed their declared identity/census gates, all
integer outputs matched a signed-128 CPU oracle, and all 320 timed outputs
were verified.

The substantial shapes favored BF16 by 2.44--3.68x in the captured
same-regime medians. The valid conclusion is STOP/PARK for this exact
unpacked integer lane, not an integer-GPU advantage. See
[`GPU_REAL_WEIGHT_RESULTS.md`](GPU_REAL_WEIGHT_RESULTS.md).

This does not replace the earlier P13/P13B fair CUDA-core baseline in the
public repository. That experiment held width, kernel structure and hardware
path constant and measured FP32 versus INT32 Q16.16 at 0.98--1.00x parity.
The combined GPU result remains parked; see
[`GPU_RESULTS.md`](GPU_RESULTS.md).

## Packed CUDA representation transferability

The first signed-int16 resident-weight candidate preserved exact integer
outputs and improved the two large pressure-labelled single-launch medians by
35.2--38.9%, but its full frozen gate remained `STOP/PARK` because seven
samples in a separate `repeat_hot` regime exceeded the stability envelope.

A separately frozen follow-up changed the measurement unit rather than the
threshold. It timed 12 GEMVs over 12 distinct resident weight copies per lane;
even the packed image was 2.75x reported L2. Across gate/up and down projection
and three fresh processes:

- packed sequence medians were 41.2--42.1% below matched wide-int32 medians;
- all 72 paired observations passed, with reductions from 40.26% to 43.16%;
- every packed and wide sample stayed within 10% of its lane median;
- every checked integer output was exact;
- SASS, sanitizer, qualification, process, pressure, and evidence-integrity
  gates passed.

This confirms a synthetic resident-representation/bandwidth signal. It does
not demonstrate an integer-arithmetic, real-weight, layer, token, or runtime
advantage. The old failed single-launch gate remains failed, and GPU work is
parked; see [`P13_ROTATED_RESULTS.md`](P13_ROTATED_RESULTS.md).

## MCU transfer outcome

The complete uniform-F12 MicroGPT workload was subsequently reproduced on a
Seeed XIAO RP2040, both ARM and no-FPU RISC-V modes of a Raspberry Pi Pico 2,
an Arduino MKR Zero, and a no-FPU ESP32-C6. Packed signed-int16 weights matched
the rounded-wide raw-logit oracle on every board. Flash use fell on all five
configurations; runtime improved materially on four and was effectively tied
on the MKR Zero. See [`MCU_RESULTS.md`](MCU_RESULTS.md).
