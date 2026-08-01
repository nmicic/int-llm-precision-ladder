# Full uniform-F12 MicroGPT result on XIAO RP2040

**Disposition: PASS — retain as a one-board representation-plus-execution
signal.** Direct signed-int16 F12 storage reproduced the rounded-wide F12
oracle exactly and was materially smaller and faster on this Cortex-M0+
board without an FPU.

| Metric | Wide uniform-F12 Q16.48 | Direct packed F12 | Difference |
|---|---:|---:|---:|
| Model container | 115,576 B | 29,944 B | −85,632 B (−74.09%) |
| Complete linked flash | 189,092 B | 103,460 B | −85,632 B (−45.29%) |
| Static RAM | 19,520 B | 19,520 B | unchanged |
| Median full 20-name run | 6,714,832 µs | 5,550,795 µs | −17.335% (1.210× throughput) |

Correctness was stronger than sample-only parity:

- native-`__int128` and forced-portable host runs produced identical
  transcripts;
- wide and packed host lanes matched all 20 names, all 122 inference steps,
  and the canonical hash of every raw pre-temperature logit word;
- both warmups and all eight physical calls in schedule `WPPWPWWP` matched
  those host pins;
- one `+1` code change in the BOS embedding left the names unchanged but
  changed the raw-logit hash, and both host and board rejected it as the
  positive result.

Timing was exceptionally stable: the wide call range was 6,714,503–6,714,979
µs (0.0071% of its median), and packed was 5,550,495–5,551,037 µs (0.0098%).
Loading/reset was outside timing. Raw-logit observation and sample hashing
were inside both timed lanes equally.

The packed firmware contains direct signed-halfword loads. It does not expand
the model into int64 RAM; activations and results remain Q16.48. The observed
speed combines narrower XIP storage/cache traffic and packed execution—it is
not an isolated arithmetic-speed claim. The reported 133 MHz clock is the
locked firmware configuration, not an independent clock measurement.

This result says one complete tiny integer model benefits on one RP2040. It
does not establish a floating-point comparison, training speedup, a board
matrix, or transfer to TinyLlama. The next experiment should remain small and
test transferability rather than immediately building a general runtime.
