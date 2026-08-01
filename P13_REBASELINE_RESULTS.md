# P13/P13B CUDA-core rebaseline

Run date: 2026-07-30
Target: NVIDIA GeForce RTX 5090, compute capability 12.0
Toolchain: CUDA 13.3, nvcc 13.3.73, driver 610.43.02

## Outcome

**VALID baseline reproduction. FP32 and INT32 Q16.16 remain at parity in the
inherited simple CUDA-core kernels.**

The unchanged P13/P13B programs were built once and captured in seven
independent process blocks with chronologically alternating executable order.
Every expected row, exit record, ratio, drift value and retained hash passed
the local audit.

No candidate or optimization result is included here.

## P13 M=1 medians

Seven observations per lane and shape:

| Shape | FP32 median us | FP32 min--max | INT32 median us | INT32 min--max | FP32/INT32 median |
|---|---:|---:|---:|---:|---:|
| q/o projection | 92.3 | 92.3--92.8 | 92.8 | 92.8--93.2 | 0.995 |
| KV projection | 92.1 | 92.1--92.6 | 92.6 | 92.5--93.0 | 0.995 |
| gate/up projection | 92.3 | 92.3--92.8 | 92.8 | 92.8--93.2 | 0.995 |
| down projection | 252.9 | 252.8--254.2 | 253.1 | 253.1--254.4 | 0.999 |

The largest observed max drift was `4.10e-4`, identical to the inherited
fixture result.

## P13B M=1 medians

P13B's different launch geometry remains the better absolute incumbent for
its two shapes:

| Shape | FP32 median us | FP32 min--max | INT32 median us | INT32 min--max | FP32/INT32 median |
|---|---:|---:|---:|---:|---:|
| q/o projection | 52.6 | 52.6--52.8 | 53.1 | 53.0--53.2 | 0.991 |
| down projection | 143.8 | 143.4--144.0 | 146.0 | 145.5--146.1 | 0.985 |

Across every P13B shape and M=1--64 observation, displayed per-run ratios
ranged from `0.982` to `1.011`. No sustained integer advantage appeared.

## Comparison with the March record

The current toolchain is modestly faster in absolute time, but the conclusion
is unchanged:

- P13 q/o INT32: 98.3 to 92.8 us;
- P13 down INT32: 255.3 to 253.1 us;
- P13B q/o INT32: 57.3 to 53.1 us;
- P13B down INT32: 149.5 to 146.0 us.

Because both lanes moved together, this is a rebaseline—not an integer
improvement.

## Provenance and instruction boundary

| Artifact | SHA-256 |
|---|---|
| P13 source | `a675b8fec20c79ac6b4c3681834c3483874b295e1ecec9f80f82ce7fd54733a7` |
| P13B source | `2d71f8f68a8f139c8abe0638943018bbc29bdeda5e0346888ae753e11286d19d` |
| P13 executable | `5fbc613f5cff6df76838ebc7d806cc6286a82a9ef55c19c0b0236a2a4b18b328` |
| P13B executable | `5baa346ff21889e87b93e7ae73ad08e1bfbcbf1d940b916af4e74a9c7a157e9b` |
| preflight manifest | `1ba66d8e7c8e2b81cb07ac4e7bf7b36ee1893579042e248eec833aee43f6f248` |
| retained manifest | `d95afd8152460b1f8dfb5975b7cea000db1ff7aa5e9f950267e36f599fe27daf` |
| accepted schedule | `41d0b9982150f506fe89e0c1a38e41fb456bb8b2d41ee0744c10c5b36b860d41` |

SASS inspection found the intended FP32 and integer kernel symbols and
positive `FFMA`/`IMAD` instruction sentinels. It found zero
`HMMA`/`IMMA`/`BMMA`/`MMA.*` tensor-matrix instructions.

## Failed pre-captures

Two pre-captures were rejected without changing the accepted evidence:

1. The first SASS command used an unqualified `cuobjdump` outside the remote
   `PATH`. The gate stopped before timing; the diagnostic is retained.
2. The first seven-block shell command labelled odd/even blocks correctly but
   executed all odd blocks before all even blocks. Those logs are retained as
   INVALID, and the accepted sequence was rerun chronologically from block 1
   through 7.

## Boundary and next question

These are deterministic synthetic fixtures and simple, memory-latency-limited
kernels. They do not measure a layer, token generation or full inference.

The frozen incumbent for a bounded next spike is:

- unchanged P13/P13B FP32 control;
- unchanged P13/P13B INT32 Q16.16 control;
- at most one independently selected integer candidate.

The candidate must materially lower the INT32 lane, preserve its declared
machine-output oracle and include any representation/decode cost. If width
changes, the result is representation plus execution—not an
integer-arithmetic-only win.

Machine-local logs, SASS, binaries, and invalid pre-captures are not part of
the public tree. The result summary and CUDA reference source are retained.
