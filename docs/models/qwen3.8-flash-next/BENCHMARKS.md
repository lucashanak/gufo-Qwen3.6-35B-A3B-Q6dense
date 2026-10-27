# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target
with shared-Q8_0 MTP; adaptive drafting. HTTP, greedy, thinking off.
Gufo `fec4759d` (October 18, 2026); llama.cpp `b11069` AR and `6fcaa16f` MTP
(September 22–23). Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1801.01 | 489.59 | +267.9% | 28.29 | 22.20 | +27.4% |
| 4,096 | 1727.95 | 455.24 | +279.6% | 28.27 | 21.21 | +33.3% |
| 8,192 | 1721.41 | 428.85 | +301.4% | 28.27 | 20.40 | +38.6% |
| 12,288 | 1710.64 | 400.21 | +327.4% | 28.21 | 19.64 | +43.6% |
| 16,384 | 1706.86 | 375.89 | +354.1% | 28.25 | 18.95 | +49.1% |
| 32,768 | 1672.60 | 301.31 | +455.1% | 28.18 | 16.54 | +70.4% |
| 65,536 | 1602.93 | 221.94 | +622.2% | 28.09 | 11.62 | +141.7% |
| 131,072 | 1557.18 | 144.78 | +975.5% | 27.85 | 7.98 | +249.0% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1734.96 ± 0.85 | 468.76 | +270.1% | 35.79 | 31.73 | +12.8% | 116.93 ± 0.10 | 48.12 | +143.0% |
| 4,096 | 1676.83 ± 1.73 | 423.79 | +295.7% | 37.64 | 34.48 | +9.2% | 116.41 ± 0.07 | 45.71 | +154.7% |
| 8,192 | 1669.91 | 394.98 | +322.8% | 36.91 | 32.20 | +14.6% | 114.08 ± 0.01 | 44.31 | +157.5% |
| 12,288 | 1660.54 ± 1.22 | 365.91 | +353.8% | 39.28 | 32.32 | +21.5% | 105.48 ± 0.15 | 43.53 | +142.3% |
| 16,384 | 1635.18 | 341.98 | +378.2% | 40.54 | 32.07 | +26.4% | 113.32 ± 0.23 | 43.66 | +159.6% |
| 32,768 | 1618.63 | 277.61 | +483.1% | 39.97 | 25.41 | +57.3% | 105.38 ± 0.17 | 39.28 | +168.3% |
| 65,536 | 1556.32 | 208.04 | +648.1% | 39.23 | 19.65 | +99.6% | 104.78 ± 0.21 | 27.75 | +277.6% |
| 131,072 | 1481.19 | 135.42 | +993.8% | 39.87 | 14.63 | +172.5% | 108.37 | 20.96 | +417.0% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 28.24 | 22.34 | +26.4% |
| 2 | 50.66 | 37.08 | +36.6% |
| 4 | 86.01 | 54.82 | +56.9% |
| 6 | 108.34 | 65.34 | +65.8% |
| 8 | 125.48 | 68.79 | +82.4% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 35.74 | 31.77 | +12.5% | 116.50 | 46.93 | +148.2% |
| 2 | 61.30 | 42.79 | +43.3% | 165.50 | 52.57 | +214.8% |
| 4 | 88.77 | 48.78 | +82.0% | 218.12 | 48.23 | +352.2% |
| 6 | 109.74 | 52.59 | +108.7% | 231.85 | 48.51 | +377.9% |
| 8 | 127.02 | 61.92 | +105.1% | 245.81 | 58.13 | +322.9% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 16.42 | 117.83 | +617.6% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 88.78 | 84.84 | -4.4% |
| 16K prefix, pp4096 + tg128 | 90.53 | 85.31 | -5.8% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

## Lower-precision MTP sidecar (October 27, 2026)

`fn-c8` `9e23078a` (`fn-mtp` `c20af263` for the K-quant attention arm,
whose Q8_0 control is identical), v3 target, greedy unless noted.
C1: five 384-token prompts, one session, 65,536 context (mean of five
decode rates). Slice: eight sessions at production flags. `fnab`: 4 ×
40,960, one round each; C1/C4/C8 are sum of rates, wall aggregate in
brackets. Rejected; see
[experiments](EXPERIMENTS.md#lower-precision-mtp-sidecar-october-27-2026).

| Measure (tok/s) | Q8_0 | Q5_K/Q5_1 experts | Q4_K/IQ4_NL experts | Q3_K/IQ4_NL experts | Q4_K experts + attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| HTTP C1 MTP, five prompts | 70.63 | 71.10 | 71.32 | 70.90 | 71.40 |
| C1 acceptance (accepted / proposed) | 69.4% | 68.8% | 68.8% | 69.1% | 68.7% |
| HTTP C1 MTP after 32.9K, 256 tokens | 52.40 / 52.03 | 51.95 | 49.47 | 47.90 | 44.83 |
| `fnab` MTP decode 512 (prose) | 61.9 | 61.8 | 62.0 | 59.2 | 61.1 |
| `fnab` code at 0 / 0.7 | 81.8 / 85.9 | 81.1 / 84.3 | 80.8 / 83.9 | 84.5 / 83.6 | 80.9 / 81.2 |
| `fnab` C1 / C4 / C8 | 59.7 / 100.3 (92.6) / 193.4 (92.4) | 57.5 / 96.2 (91.6) / 197.1 (92.7) | 62.3 / 100.6 (93.8) / 194.5 (92.5) | 61.1 / 99.4 (93.7) / 196.8 (93.3) | 57.6 / 102.1 (98.8) / 193.8 (93.8) |
| 28-question slice wall (s) / rounds | 333.6 / 8,124 | | 333.0 / 8,460 | | 334.1 / 8,604 |

Uncensored target, Q8_0 → Q4_K/IQ4_NL experts: C1 59.61 → 59.54 (5/5
identical); `fnab` prose 51.5 → 49.0, code at 0 / 0.7 74.0 / 73.4 →
75.1 / 73.6, C1 / C4 / C8 50.4 / 86.3 / 169.0 → 50.4 / 87.6 / 172.1.

## Single-user decode (October 27, 2026)

v3 recipe + shared-Q8_0 MTP, greedy, thinking off. Baseline fn-c8
`bcac0128`, candidate the single-user decode changes on top of it
([experiments](EXPERIMENTS.md#single-user-decode-october-27-2026)). The
HTTP rows use `gufo serve` with one session and 65,536 context: five fixed
prompts of 384 tokens (mean of the five decode rates) and one 32,879-token
prompt; outputs are byte-identical between the arms.

| Measure (tok/s) | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| CLI AR tg256 after pp512, two runs | 39.03 / 39.64 | 41.20 / 41.13 | +4.6% |
| CLI MTP tg256 after pp512, two runs | 71.74 / 71.74 | 72.21 / 72.83 | +1.1% |
| HTTP C1 AR, five prompts | 38.82 | 40.83 | +5.2% |
| HTTP C1 MTP, five prompts | 69.53 | 70.77 | +1.8% |
| HTTP C1 AR after 32.9K, 256 tokens, alternating | 35.17 / 34.79 | 39.15 / 39.36 | +12% |
| HTTP C1 MTP after 32.9K | 52.13 | 53.49 | +2.6% |

`fnab` (4 × 40,960, one round each), baseline → candidate: pp8192
1,620 → 1,600 and pp32768 1,627 → 1,595 (prefill code unchanged);
MTP decode 512 (prose) 60.9 → 62.9; code at 0 / 0.7 79.4 / 82.7 →
81.2 / 80.1; C1 / C4 / C8 MTP (sum of rates) 62.6 / 100.9 / 195.0 →
61.6 / 101.4 / 191.3; C4 / C8 wall aggregate 95.2 / 92.9 → 92.3 / 91.3.
The eight-way 28-question slice at production flags takes 333.3 s
(fn-c8 333.9 s) with all 28 answers identical.

## Eight-way suite replay (October 27, 2026)

The swapctl IQ suite (149 questions, `core` tier) replayed greedy by eight
workers at the production flags: v3 recipe + shared-Q8_0 MTP, 8 × 262,144,
`--kv-cache-bytes 21474836480`, `--reasoning-budget-fraction 0.7`,
`--cache-ram-bytes 4294967296`. Baseline `c9b4fc82` (fn-kvmem + upstream
0.25.0); candidate `fn-c8` on the same base. Prefill is the sum of each
request's prompt tokens over its prefill rate; decode is the rest of the wall.

| Measure | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Wall (s) | 1,716.7 | 1,546.9 | −9.9% |
| Generated tokens | 147,180 | 147,686 | |
| Aggregate (tok/s) | 85.7 | 95.5 | +11.4% |
| Prefill (s, 428,563 prompt tokens) | 274.8 | 274.6 | 0% |
| Wall minus prefill (s) | 1,441.9 | 1,272.3 | −11.8% |
| 28-question slice wall (s) / aggregate (tok/s) | 366.1 / 77.6 | 333.9 / 85.1 | −8.8% / +9.7% |

`fnab` (4 × 40,960, one round each), tok/s:

| Measure | Baseline | Candidate |
| --- | ---: | ---: |
| pp8192 / pp32768 | 1,623 / 1,603 | 1,612 / 1,602 |
| MTP decode 512 (prose) | 60.2 | 58.9 |
| MTP code at 0 / 0.7 | 78.7 / 75.8 | 80.0 / 83.3 |
| C1 / C4 / C8 MTP (sum of rates) | 58.6 / 84.3 / 181.0 | 58.9 / 101.3 / 193.7 |
| C4 / C8 wall aggregate | 80.2 / 87.1 | 95.7 / 90.9 |

## K/V memory (October 25, 2026)

v3 recipe + shared-Q8_0 MTP, `serve` with `--cache-ram-bytes 4294967296`.
GPU memory from the loader's `gpu_device_used_mib` at readiness; host
`MemAvailable` from the Proxmox host (the serving floor is 12 GiB).

| K/V | Sessions × context | GPU used at ready (MiB) | Host available at ready (GiB) |
| --- | --- | ---: | ---: |
| F16, preallocated (`366b9c09`) | 8 × 40,960 | 91,199 | 23 |
| 8-bit, preallocated | 8 × 40,960 | 87,871 | 28 |
| 8-bit, growing | 8 × 40,960 | 82,463 | 34 |
| 8-bit, growing | 8 × 262,144 | 82,463 | 34 |
| 8-bit, growing, uncensored GGUF | 8 × 262,144 | 87,702 | 32 |
| F16, preallocated | 1 × 262,144 | 86,861 | — |
| 8-bit, preallocated | 2 × 262,144 | 88,283 | 22 |

With growing rows, eight concurrent 166K-token needle requests on
8 × 262,144 (1.33M tokens held) answered 8/8 with host `MemAvailable` at
least 14 GiB: about 15.6 KiB per held token including chunk rounding (8-bit
K and V, pooled keys and draft rows; F16 needs about 27 KiB). The uncensored
GGUF leaves about 20 GiB above the floor at readiness, room for about
1.3M held tokens across its sessions; `--kv-cache-bytes` should cap it there.
The F16 build at 2 × 262,144 did not reach readiness.

Speed, alternating rounds (F16 `366b9c09`; 8-bit preallocated; 8-bit
growing with the stream drain later removed), `fnab` at 4 × 40,960 and C1 at
depth on 2 × 163,840, tok/s:

| Measure | F16 | 8-bit preallocated | 8-bit growing |
| --- | ---: | ---: | ---: |
| pp8192 | 1,651 / 1,633 | 1,612 / 1,685 | 1,622 / 1,625 |
| pp32768 | 1,622 / 1,604 | 1,597 / 1,593 | 1,597 / 1,596 |
| MTP decode 512 (prose) | 58.9 / 56.8 | 58.1 / 58.9 | 59.1 / 60.3 |
| MTP code at 0 / 0.7 | 78.2, 76.3 / 78.0, 78.3 | 77.8, 80.0 / 77.3, 77.8 | 78.7, 77.5 / 78.7, 77.9 |
| C1 / C4 / C8 MTP (sum of rates) | 60.3, 87.6, 177.3 / 58.6, 93.0, 180.0 | 52.4, 88.7, 176.7 / 58.1, 88.3, 177.0 | 53.7, 88.2, 165.7 / 58.2, 88.3, 180.8 |
| AR prefill 25.7K / 102.6K | 1,727, 1,639 / 1,694, 1,645 | 1,685, 1,644 / 1,679, 1,679 | 1,678, 1,601 / 1,675, 1,632 |
| AR decode after 25.7K / 102.6K | 37.75, 36.88 / 37.80, 36.85 | 37.83, 36.82 / 37.75, 37.08 | 37.62, 36.73 / 37.17, 37.29 |
| 8-way replay, longctx + exec (40 requests) | 73.4 / 73.0 aggregate | 76.7 / 76.6 | 76.5 / 76.6 |

Decode is unchanged. Prefill is within about 1.5% (32K) and noisy at 128K;
in kernel traces the prefill kernels sum to −0.6% (25.7K) and +0.3% (102.6K)
against F16, the only systematic cost being sparse WMMA attention (+3.6% /
+2.6% per call). The replay aggregates differ partly by output mix (KV8
answers held more tokens); per-request decode rates are equal.

