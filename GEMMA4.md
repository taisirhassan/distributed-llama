# Gemma 4 (dense text) support — design note

Target: `google/gemma-4-12B-it` (48 layers, dim 3840, ff 15360, vocab 262144).
Reference implementations checked: HF `transformers` `modeling_gemma4.py` /
`configuration_gemma4.py` / `modeling_rope_utils.py` (main, 2026-09) and
`ggml-org/llama.cpp` `src/models/gemma4.cpp` + `conversion/gemma.py`.

## Architecture facts that matter (verified in the reference code)

| Item | Gemma 4 12B | Where it is handled |
|---|---|---|
| RMSNorm | `x * rsqrt(mean(x^2)+eps) * (1 + w)` | converter: Google's GGUF already stores `1 + w`; a safetensors converter must add 1 |
| Embedding scale | `embed(x) * sqrt(dim)` (f32 61.9677; HF rounds it to bf16 = 62.0, llama.cpp uses f32) | converter folds `sqrt(dim)` into the F32 embedding table |
| Tied LM head | `lm_head = embed_tokens` (unscaled) | converter writes `wcls` from the embedding table as q40 |
| Block norms | `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm` | engine: two extra RMS norms per block, applied on the *summed* (cross-node) attention / FFN output before the residual add |
| QK norm | `q_norm`, `k_norm` per head (with weight) | engine: reuse the Qwen3 path |
| V norm | `v_norm` = RMSNorm **without** weight, on every layer | engine: RMS norm with a constant weight of ones written by the converter (`block_norm_v`) |
| Attention scale | `scaling = 1.0` (no `1/sqrt(head_dim)`) | engine: `NnMultiHeadAttOpConfig.scale` |
| Layer types | 5 × `sliding_attention` then 1 × `full_attention` (layers 5, 11, …, 47 are full) | header `FULL_ATT_INTERVAL = 6` |
| Sliding layers | head_dim 256, 16 q heads, 8 kv heads, window 1024 (`q - k < 1024`), rope theta 1e4, full rotary | header `SLIDING_WINDOW`, `ROPE_THETA_SWA`; engine: windowed mask in `OP_MULTIHEAD_ATT` |
| Full layers | head_dim **512**, 16 q heads, **1** kv head, causal, rope theta 1e6, "proportional" partial rotary 0.25 → only the first 64 of 256 rotation pairs rotate, exponent uses `2i/512` | header `HEAD_DIM_FULL`, `N_KV_HEADS_FULL`, `ROPE_DIMS_FULL = 128`; engine: per-type rope cache, `NnRopeSlice.ropeDims` |
| `attention_k_eq_v` | full layers have **no `v_proj`**: `V = v_norm(k_proj(x))` (pre-`k_norm`) | engine: copy `k_temp` to `v_temp` before the K norm on full layers; converter writes no V for those layers |
| Activation | `gelu_pytorch_tanh` | existing `OP_GELU` (tanh approximation) |
| Final logits | `30 * tanh(logits / 30)` | header `FINAL_LOGIT_SOFTCAP`; engine `OP_SOFTCAP` |
| `layer_scalar` | `hidden_states *= layer_scalar` at the end of every block (GGUF `layer_output_scale`, values 0.0037 to 0.9 in the 12B, so it cannot be dropped) | engine `OP_SCALAR_MUL` with a 1-element weight, applied after the post-FFN residual add |
| Per-layer embeddings, MoE, kv-shared layers | not present in 12B (`hidden_size_per_layer_input = 0`, `num_kv_shared_layers = 0`) | rejected by the converter |
| Rope style | NeoX / `rotate_half` (pairs `(j, j + head_dim/2)`) | existing `ROPE_FALCON` kernel |

## Multi-node slicing

Sliding layers slice like Qwen3 (q/k/v row-sliced, `wo` column-sliced, each node owns
`nKvHeads / nNodes` kv heads). Full layers have a single kv head, so K cannot be
row-sliced across nodes: the K projection is loaded on every node with `loadAll`
(3840 × 512 q40 = 1.1 MB per layer) and every node computes the full K/V head
redundantly; the KV cache for those layers is not sliced (`kvDim0 = kvDim`). The
general rule used by the builder: replicate KV whenever `nKvHeads % nNodes != 0`.

The post-attention / post-FFN norms need the full cross-node sum before the residual
add, so a new `OP_MERGE_SET` (output = sum of node slices, no residual) feeds
`OP_INV_RMS` + `OP_RMS_NORM`, and the existing `OP_MERGE_ADD` (with a single
slice) adds the normed result to the residual stream.

## New header keys (`src/llm.hpp`, `converter/writer.py`)

| Key | Id | Value for 12B |
|---|---|---|
| `SLIDING_WINDOW` | 22 | 1024 |
| `FULL_ATT_INTERVAL` | 23 | 6 (layer `l` is full iff `(l + 1) % 6 == 0`; 0 = no sliding layers) |
| `ROPE_THETA_SWA` | 24 | 10000 (`ROPE_THETA` = 1000000 is the full-layer theta) |
| `ROPE_DIMS_FULL` | 25 | 128 (rotated dims of the 512-dim full-layer head) |
| `HEAD_DIM_FULL` | 26 | 512 |
| `N_KV_HEADS_FULL` | 27 | 1 |
| `FINAL_LOGIT_SOFTCAP` | 28 | 30 (0 = none) |

Arch type: `GEMMA4 = 0xABCD03`.

## Weight file layout written by the converter (per layer, in order)

`q`, `k`, [`v` (sliding layers only)], `wo`, `w1` (gate), `w2` (down), `w3` (up),
`q_norm`, `k_norm`, `v_norm` (ones), `norm_0` (input), `norm_post_att`,
`norm_1` (pre-ff), `norm_post_ff`, `layer_scale` (1 float); then `final_norm`, `wcls`.

## Source of weights

`google/gemma-4-12B-it-qat-q4_0-gguf` (not gated). ggml `block_q4_0`
(`fp16 d; uint8 qs[16]`, low nibble = element `j`, high nibble = `j + 16`,
`(q - 8) * d`) is byte-identical to dllama's `NnBlockQ40`, so all Q4_0 matrices
are passed through untouched. `token_embd.weight` is **Q6_K** in that file; the
converter dequantizes it (`gguf.quants`) to F32 for the embedding and requantizes
it to Q4_0 for the tied LM head (a lossy step llama.cpp does not take: it uses the
Q6_K table directly as the output matrix).

## What `google/gemma-4-26B-A4B-it` would need on top of this (not implemented)

Read from `modeling_gemma4.py` (`Gemma4TextDecoderLayer`, `Gemma4TextRouter`,
`Gemma4TextExperts`) and `llama.cpp/src/models/gemma4.cpp` (`is_moe_layer` branch):

* Same attention scheme (sliding/full pattern, `attention_k_eq_v`, QK/V norms,
  scale 1.0), so every header key above is reused unchanged; the only attention
  difference is 8 kv heads on sliding layers and 2 on full layers
  (`num_global_key_value_heads = 2`), which the "replicate KV when
  `nKvHeads % nNodes != 0`" rule already covers.
* Every MoE layer runs **both** a dense MLP and the expert MLPs on the same input
  and sums them:
  `h1 = post_ffn_norm_1(dense_mlp(pre_ffn_norm(x)))`,
  `h2 = post_ffn_norm_2(experts(pre_ffn_norm_2(x), topk))`,
  `out = post_ffn_norm(h1 + h2)`, then the residual add. So three extra norms per
  layer (`ffn_pre_norm_2`, `ffn_post_norm_1`, `ffn_post_norm_2`) and the dense
  `intermediate_size = 2112` alongside `moe_intermediate_size = 704`.
* Router (`Gemma4TextRouter`) is *not* the Qwen3 gate: it takes the residual stream
  `x` (before `pre_ffn_norm`), applies a weight-less RMSNorm, multiplies by a learned
  per-dim `scale` and by `1/sqrt(dim)`, projects to 128 logits, softmax in fp32,
  top-8, renormalises the top-k weights to sum to 1 (the existing
  `OP_MOE_GATE{normTopk = 1}` does this part), then multiplies each weight by a
  learned `per_expert_scale[expert]` (llama.cpp loads it as `down_exps_s`). The
  engine would need the extra per-dim scale + per-expert scale inputs.
* Experts are stored fused as `gate_up_proj[E, 2*704, dim]` in HF (llama.cpp splits
  them into `ffn_gate_exps` / `ffn_up_exps` at conversion; the QAT GGUF may keep them
  fused as `ffn_gate_up_exps`); activation is GELU-tanh, not SiLU.
* Header additions that would be needed: `MOE_HIDDEN_DIM` already exists; add a
  `MOE_LAYER_PATTERN`/flag if not every layer is MoE (in 26B-A4B every layer is,
  per llama.cpp's `is_moe_layer = ffn_gate_inp != nullptr`), and reuse
  `N_EXPERTS` / `N_ACTIVE_EXPERTS`.

## Validation (MacBook Pro M3 Pro, 19 GB RAM, CPU)

`google/gemma-4-12B-it-qat-q4_0-gguf` converted with `converter/convert-gguf.py`
(10,727,120,256 bytes: 4.0 GB F32 embedding + 6.7 GB Q4_0), tokenizer from the HF
`tokenizer.json`. llama.cpp (a894dae, CPU, greedy) on the same GGUF versus
`dllama inference --temperature 0 --buffer-float-type q80 --max-seq-len 512`, three
chat prompts, comparing generated tokens up to llama.cpp's end-of-turn:

| Prompt | llama.cpp tokens | identical |
|---|---|---|
| "What is the capital of France? Answer in one sentence." | 8 | 8/8 |
| "Write a haiku about the ocean." | 21 | 21/21 |
| "List three prime numbers greater than 10 and explain briefly why they are prime." | 48 | 48/48 |

Token-by-token match rate against llama.cpp: 77/77 generated tokens identical (100%)
across the three prompts (the longest common prefix equals the full llama.cpp output in
every case). Prompt tokenization is identical to HF `tokenizers` for these prompts
(and for ASCII/code text in general; non-ASCII text falls back to byte tokens in
dllama's encoder, see below). `dllama chat` with the built-in Gemma 4 template answers
"Name three colors of the rainbow, comma separated." with "Red, orange, yellow".

Throughput on this Mac (M3 Pro, 11 cores, 1 node, `--nthreads 8`, 128 generated
tokens, `--max-seq-len 512`): 4.74 tokens/s generation (211 ms/token), 12.3 tokens/s
prompt evaluation (nBatches 32). With `--nthreads 10` it drops to 2.0 tokens/s. The
machine has 19 GB RAM and was swapping (the process needs the 4 GB F32 embedding
table + 6.7 GB Q4_0 weights + KV cache), so these are lower bounds. For reference,
llama.cpp's `llama-simple` on the same GGUF on the same machine decoded at 0.9 to
4.3 tokens/s under the same memory pressure.

### Multi-node (localhost workers on the same Mac, root `--nthreads 4/3`, workers 2/1 threads)

Same three greedy prompts, 48 generated tokens each, compared token by token with the
1-node output (`dllama worker --port 999N` + root `--workers 127.0.0.1:9991 ...`). The
full-attention layers replicate their single KV head on every node whenever the node
count does not divide 1, so the replicated-K path (`loadAll` + `qHeadOffset`) runs in
every multi-node configuration:

| Nodes | Prompt 1 | Prompt 2 | Prompt 3 | generation tokens/s (per prompt) |
|---|---|---|---|---|
| 2 | 48/48 | 48/48 | 48/48 | 4.1 / 3.7 / 4.2 |
| 4 | 35/48 (identical through the answer; diverges after the end-of-turn token) | 48/48 | 13/48 (" a brief explanation" vs " the reasons", both coherent) | 3.8 / 2.7 / 4.2 |
| 8 | 48/48 | 48/48 | 48/48 | 0.3 (swapping) / 4.0 / 2.9 |

The 4-node divergence is the known effect of dllama's Q80 node sync: every node
quantizes its partial attention/FFN output to Q80 before the cross-node sum, so the
residual stream differs by node count and greedy decoding flips near-tied tokens. The
control with the existing Qwen3 0.6B q40 model on the same user text shows the same
behaviour (1 vs 2 nodes: 70/70 identical; 1 vs 4 nodes: identical for 41 tokens, then
" number" vs " natural"), and 8 nodes reproduce the 1-node Gemma output exactly, so
the slicing itself is consistent. The per-token sync cost on localhost with 8 nodes was
about 140 ms (2.8 MB sent / 3.6 MB received per token) versus 73 ms of compute.

Converted files (not committed, `/models` is git-ignored):

```
models/gemma4_12b_q40/dllama_model_gemma4_12b_q40.m   10,727,120,256 bytes
models/gemma4_12b_q40/dllama_tokenizer_gemma4_12b_q40.t        4,155,012 bytes
```

Reproduce (from the repository root, Python env with `gguf` and `numpy`):

```sh
make dllama dllama-api nn-cpu-test && ./nn-cpu-test          # 16 kernel tests
# inputs: gemma-4-12b-it-qat-q4_0.gguf (google/gemma-4-12B-it-qat-q4_0-gguf) and
# config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja (google/gemma-4-12B-it)
python converter/convert-gguf.py <dir>/gemma-4-12b-it-qat-q4_0.gguf gemma4_12b_it
python converter/convert-tokenizer-hf.py <dir> gemma4_12b_it
mkdir -p models/gemma4_12b_q40 && mv dllama_model_gemma4_12b_it_q40.m models/gemma4_12b_q40/dllama_model_gemma4_12b_q40.m && mv dllama_tokenizer_gemma4_12b_it.t models/gemma4_12b_q40/dllama_tokenizer_gemma4_12b_q40.t
./dllama inference --model models/gemma4_12b_q40/dllama_model_gemma4_12b_q40.m \
  --tokenizer models/gemma4_12b_q40/dllama_tokenizer_gemma4_12b_q40.t \
  --buffer-float-type q80 --nthreads 8 --max-seq-len 512 --temperature 0 --steps 73 \
  --prompt "$(printf '<|turn>user\nWhat is the capital of France? Answer in one sentence.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>')"
```

The three validation prompts use that exact chat wrapping (the canonical template with
thinking disabled appends an empty `<|channel>thought\n<channel|>` block) with the user
texts listed in the table; llama.cpp was run with `llama-simple`-style greedy sampling
(`llama_sampler_init_greedy`, `llama_tokenize(..., add_special=true, parse_special=true)`)
on the same prompt strings. `nn-cpu-test` covers the windowed attention (incl. the
replicated-KV head offset), partial rotary rope, softcap, scalar multiply and merge-set
kernels against straightforward references.

## Not supported

* Vulkan for GEMMA4 (`OP_SOFTCAP`, `OP_MERGE_SET`, `OP_SCALAR_MUL`, `OP_GELU` have no
  shaders; the app refuses `--gpu-index` for this arch).
* Multimodal (vision/audio towers are skipped), MoE variants (26B-A4B), per-layer
  embeddings (E2B/E4B).
* Tokenizer: dllama's greedy best-score merge is driven by scores derived from the
  BPE merge ranks; multi-byte UTF-8 characters are split into byte tokens by the
  existing encoder (pre-existing limitation).
