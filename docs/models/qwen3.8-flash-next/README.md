# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards);
standard llama.cpp mixes load too (see [GGUF formats](#gguf-formats)).
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) · [ROCm 10.1 status](ROCM.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
nix build
MODEL=/path/to/first-target-shard.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

The loader discovers the remaining shards. Omit the speculative options for AR;
AR sessions allocate no predictor state even if a shared model has MTP loaded.
Adaptive MTP is default, with `--draft-tokens` capping 1–7 proposals. Eligible
requests draft over the first 98,304 token IDs plus a 1,024-row table containing
special tokens and other tokens already used by the request. The full head
remains the fallback when the anchor or two of the last 32 committed tokens
fall outside this vocabulary.

Single-session sampling, including the default thinking-on and thinking-off
presets, can use this smaller head with the existing acceptance/length
controller and AR backoff. Greedy requests, including penalties and
constraints, can use it in C1, when decoding alone and in jointly planned
batches; drafts ignore both, and verification applies them over the full
vocabulary. Greedy requests in unplanned multi-request batches retain the full
head.

The existing length controller and AR backoff apply to every request. The
target always verifies with its full vocabulary, including
exact sampled acceptance and residual correction. Sampled text can differ
across builds when proposals change. All-greedy C>1 batches retain their shared
planner and may use measured cycle costs. Each request keeps private caches,
rollback and RNG; snapshots preserve the learned acceptance history.
See [MTP qualification](QUALITY.md).

MTP decoding also proposes copies from the context, with one to eight sessions.
The context (committed tokens plus the chain so far) is matched against the
session's own earlier tokens by its last three tokens; when the best of the 16
most recent occurrences matches at least 12 tokens back for greedy requests
or 24 for sampled requests, the tokens that followed it fill the rest of the
chain and drafting stops. Greedy requests can
bypass the neural head when a match supplies at least three copies. Sampled
requests first make a neural proposal, then look for a copy extending it;
a repeated prefix alone can poorly predict new code or field values.
Their copy sources end at the request's starting context. Earlier parts of
the same new answer are not indexed for sampled copying; they become eligible
context on the next request. Greedy requests can also copy their current answer.
The request's first speculative step skips lookup to avoid delaying its first
token and records the start of generated text. Later matches cannot extend
back into the prompt's chat wrapper; a common tool header alone must not
qualify as a long matching answer. Copies
are verified like any draft: greedy text is byte-identical, and sampled
requests treat each copied token as a point-mass proposal under the same
rejection sampling, so outputs keep the target distribution (a seeded sampled
stream differs from one without copies, because random draws are consumed at
different points). Shorter matches
propose nothing; at temperature 1 an ungated 3-token rule displaced accepted
MTP drafts and was slower on every workload measured. A recently accepted
passage can resume after an edit with a five-token match; fewer than half of
the copied tokens being accepted disables that shorter gate, including after
a full match. Copies stop at EOS when the request enables it.
The index is private to the session and rebuilt after reset or snapshot
restore. Snapshots retain the request boundary and copy-resume cursor.
Lookup cycles and their following catch-up are excluded from the
batch planner's neural cost measurements. The planner measures short neural
chains before probing longer ones after copying. In all-greedy batches mixing
copies with new text, copied suffixes are bounded to half the verifier or the
planner's chosen width, whichever is larger, to limit pauses for other requests.
All-copy batches retain full width; sampled proposal streams are unchanged.
Timings report
`lookup_n` / `lookup_n_accepted`.

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144. A larger
context needs static YaRN and `--enable-yarn`: `gufo serve llm` refuses to
start with `--context` above 262144 without it, and `gufo bench` refuses a
workload whose depth plus prompt or generation tokens exceeds 262144. With
the flag, the factor is context / 262144 (`--context 409600` gives 1.5625).
The flag is an error at or below 262144, where it would have no effect.
Static YaRN applies to every request on that server, including short ones,
and changes the output distribution more as the factor grows; in the first
measurements, multiple-choice accuracy moved within noise at factors 1.5625
to 3.875. See the
[measurements](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323981940),
the
[opt-in proposal](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323982386)
and the
[accuracy check](https://github.com/gufo-org/gufo/pull/350#issuecomment-6234093998).
No GGUF key enables YaRN.
Prefill chunks whose visible context exceeds 262144 tokens use the per-token
attention kernel rather than the fused WMMA kernel. Memory grows with used
context and selected rollback depth (about 27.5 KiB per context token with
MTP, measured on gfx1151); admission reserves the configured capacity before
creating sessions.

## GGUF formats

Every GGML block format that `llama-quantize` writes for this model loads:
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS, besides F32/F16/BF16, in the
tensor classes below. The routed experts take every format except IQ1_M,
which the tiled prefill kernels do not implement.

| Tensors | Formats | Decode | Prefill |
| --- | --- | --- | --- |
| `token_embd` | any | exact row gather (`qfn_mmq_get_rows`) | same |
| `output`, trunk, shared expert, PLE projections, `ssm_alpha`/`ssm_beta` | Q8_0, F16/BF16/F32, or any block format | the file's rows (`qfn_mmq_dense_kq_vec`) | their Q8_0 re-encoding (W8A8 / F16 GEMMs) |
| HC mixers | as above (320-wide `up`: 32-element formats); `inject` any format, widened to F32 unless Q8_0/BF16 | fused kernel for Q8_0, Q6_K/Q5_1 and IQ4_NL; others separate GEMVs | Q8_0 re-encoding |
| Routed gate/up | all but IQ1_M | Q3_K/Q4_K/Q5_K/Q8_0 gated kernel; others paired vector kernel + SwiGLU (batched steps too) | Q3_K/Q4_K/Q5_K (paired) and IQ4_XS (separate 128-token tiles): WMMA; others tiled MMQ |
| Routed down (640 wide) | 32-element formats | vector kernel | Q5_1, Q5_0, Q4_0, IQ4_NL, Q8_0: WMMA (with a WMMA gate/up); others tiled MMQ |
| `per_layer_token_embd` (160 wide) | 32-element formats, BF16 | CPU decode of disk rows | same |
| `ple_conv1d` | F32, F16, BF16 (widened to F32) | | |

The exactness contract holds for every format: a decode row is bit for bit
the same alone, in a batched step and as an MTP verification row, and a
prompt row does not depend on the pass that computes it
(`projection_ops`, `routed_wmma_ops`; [Quality](QUALITY.md#gguf-formats-october-28-2026)).
Formats outside the fast routes cost speed, not exactness; measured costs
are in [Experiments](EXPERIMENTS.md#gguf-formats-october-28-2026).

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Tools and artifacts

Model tests are in `tests/models/qwen38_flash_next`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`nix develop -c tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.
