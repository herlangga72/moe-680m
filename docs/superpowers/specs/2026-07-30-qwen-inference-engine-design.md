# Qwen 3.6 35B A3B MTP Inference Engine — Design Spec

**Target:** Ryzen 7 6800H / Radeon 680M (RDNA2, 12 CU, 27 GB DDR5, UMA)  
**Model:** Qwen 3.6 35B A3B MTP, iQ4_XS quantization  
**API:** Anthropic Messages API (Claude Code native)  
**Goal:** Interactive chat, correctness first, full Qwen 3.6 feature parity

---

## 1. Architecture

```
 Anthropic API Server (hyper, single-thread tokio)
       │ POST /v1/messages  (SSE streaming, tool use)
       ▼
 Inference Engine
       │ Tokenizer → Embed → [Layer 0..N-1] → LM Head → MTP Draft → Verify → Sample
       ▼
 Vulkan Compute (RDNA2 / RADV, ash crate)
       │ Single queue, pre-chained dispatches, fused dequant shaders
       ▼
 GGUF Model (iQ4_XS weights, pre-allocated buffer pool, unified memory)
```

Single binary. Rust + ash + gigatoken + hyper. No Python, no C++.

---

## 2. Anthropic API Surface

Only what Claude Code uses:

```
POST /v1/messages
  { model, messages, system?, tools?, max_tokens, stream, temperature?, top_p?, top_k?, stop_sequences? }
  → SSE: message_start → content_block_start → content_block_delta* → content_block_stop →
         message_delta ({stop_reason, usage}) → message_stop

POST /v1/messages/count_tokens   ← optional
```

Tool use detected via state machine in SSE stream handler (TEXT → TOOL_JSON on `{` match, parse complete JSON, emit tool_use content blocks). Stop reasons: `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`. No images, no extended thinking, no prompt caching, no auth — localhost only.

---

## 3. Inference Pipeline

### 3.1 Tokenizer

Gigatoken Rust crate. Qwen 3.6 native support. GB/s throughput. Chat template applied manually from model's `tokenizer_config.json`, compiled once at startup.

### 3.2 Prefill (first token)

```
embed[seq, dim] = Embed(token_ids)
for layer in 0..n_layers:
    h = RMSNorm(h)
    q, k, v = QKV(h) + RoPE(q, k)       ← causal mask, KV cache write
    attn_out = GQA(q, k, v, mask)
    h = ResidualAdd(h, attn_out)
    h = RMSNorm(h)
    gate_logits = Router(h)              ← top-K expert per token
    moe_out = SharedExperts(h) + MoE(h, gate_logits)
    h = ResidualAdd(h, moe_out)
logits = LMHead(RMSNorm(h))
token = Sample(logits[-1])
```

### 3.3 Decode + MTP Speculative

```
# Main model forward (1 token)
embed[1] → all layers → hidden_state, token[t+1]

# MTP draft chain
mtp_hidden = hidden_state
for i in 0..mtp_depth:
    h = concat(mtp_hidden, Embed(draft[i]))
    h = RMSNorm(h)
    h = MTPAttention(h, pos + i)          ← no causal mask
    h = RMSNorm(h)
    h = MTPSwiGLU(h)
    draft[i+1] = Sample(MTPHead[i](h))
    mtp_hidden = h

# Verify pass
full_forward([draft...])  → accept prefix of argmax-matches, reject rest
```

Depth=2, ~60-80% acceptance rate → 1.1-1.5 extra tokens per verify pass.

### 3.4 Sampler

GPU-side branchless CDF search. Top-p + top-k + temperature. Single dispatch, reads logits buffer, writes token ID.

---

## 4. Memory Budget

```
Available RAM:        27 GB
OS + driver:          ~3 GB
Usable:              ~24 GB
────────────────────────
Weights (iQ4_XS):    ~18.6 GB
KV Cache (mixed):     ~2.3 GB  (16K ctx, K=Q4_0, V=Int8)
Compute buffers:       ~1.0 GB
Gigatoken vocab:       ~0.1 GB
────────────────────────
Total:                ~22.0 GB
Headroom:              ~2.0 GB
```

### 4.1 KV Cache (Mixed Quantization)

- **K: Q4_0** — 4x bandwidth savings vs FP16 on the bottleneck path (`Q × K^T`). Dequant fused into attention shader. Q4_0 per-block (32-element) scales, error stays bounded over 16K tokens.
- **V: Int8** — accumulated values, no dot-product benefit from deeper quant. Int8 is half FP16 bytes, trivial unpack.
- Layout: `[layer][kv_head][seq_pos]`, pre-allocated pool, index by layer + head + position.

### 4.2 Compute Buffers

Pre-allocated once, reused every layer:
- `hidden_state`: dim × 4B (FP32 accumulate)
- `q, k, v`: n_heads × head_dim × 2B (FP16)
- `attn_output`: dim × 2B
- `gate_logits`: n_experts × 2B
- `logits`: vocab_size × 4B
- `intermediate`: dim × n_active × h_mult × 2B (MoE working set)

---

## 5. RDNA2 Machine Sympathy

### 5.1 Execution-Only Barriers

Within a layer, dispatches that pass data via the same buffer use **execution-only barriers** — no memory barrier, no cache flush:

```c
VkMemoryBarrier2 NO_FLUSH = {
    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    // NO VK_ACCESS_MEMORY_* — execution barrier only
};
```

The driver knows compute→compute on the same queue preserves ordering. No memory barrier means no L2 flush to system RAM. On an iGPU where L2 is tiny (512 KB) and every flush hits DDR5, this saves ~50μs per transition. 30 layers × 12 dispatches × 50μs = **~18ms saved per token**.

Full memory barriers (with cache flush) only at:
- After LM head write (CPU reads logits for sampling, or GPU-side sampler reads)
- After KV cache write completes (next token's read must see it)
- After final output token buffer write (CPU reads for SSE framing)

### 5.2 Pre-Chained Dispatch

All layer dispatches for one forward pass submitted in a single `vkQueueSubmit2` call with timeline semaphores. GPU runs the full chain without CPU involvement. CPU thread is free during GPU execution.

```
vkQueueSubmit2(
    L0 → sem[0] signal
    L1 → sem[0] wait, sem[1] signal
    ...
    LM_head → sem[N] wait, sem[N+1] signal
)
```

### 5.3 CPU/GPU Overlap

| GPU phase | CPU parallel |
|---|---|
| Prefill layers | Gigatoken next input (trivial, ~1ms) |
| Decode layers | SSE-frame previous token, stop-check |
| Verify pass | Tool-use state machine, prep next request |
| LM head + sample | Read token from unified memory (~50ns), embed for next iteration |

Unified memory means the sample-to-embed gap is a cache-line miss, not a PCIe transfer. GPU pipeline drain is ~10-20μs.

### 5.4 Workgroup Sizing

**Always 256 threads (4 wavefronts).** RDNA2 CUs run 4 wavefronts max concurrently. A 1024-thread WG serializes 16 waves onto one CU while others idle. 256 threads = exactly one CU fully utilized, other CUs free for other dispatches (or the next layer's dispatch if depth-issued).

### 5.5 Core Pinning

Server thread pinned to cores 0-3 (CCX0). GPU allowed to use CCX1's L3 for compute working set. Avoids CPU thread evicting GPU-hot cache lines. `taskset -c 0-3` or `sched_setaffinity` at startup.

### 5.6 Wave64 Subgroup Ops

Subgroup size = 64. All reductions (softmax, RMSNorm, router argmax) use `subgroupAdd`, `subgroupMax` — no shared memory, no barriers within a wavefront. RDNA2 executes these as single-cycle cross-lane ops.

### 5.7 FP16 Packed Math + Fused Dequant

`shaderFloat16` = 2× throughput. All attention and FFN compute in FP16, accumulate in FP32. Weight dequant (iQ4_XS → FP16) fused into every weight-reading shader — ~10 ALU ops per element, hidden behind memory wait.

### 5.8 Push Constants

Dimensions, strides, offsets in 128B push constants. No descriptor updates for per-layer metadata. One less buffer to bind per dispatch.

### 5.9 Memory Alignment

All buffers 128-byte aligned (LCM of GPU L2 128B line and DDR5 64B burst). Avoids misaligned double-fetch on cache line boundaries.

---

## 6. GGUF + iQ4_XS

Parse GGUF header → tensor metadata → kv pairs → tensor data. Weights stay iQ4_XS on disk and in memory. Dequant fused into matmul shaders — no separate dequant pass, no extra buffer, no extra dispatch.

iQ4_XS exact block layout depends on the GGUF quant version shipped with the model. Reference implementation is in llama.cpp `ggml-quants.c` — we port the dequant for whichever variant the model file uses. Approximate: ~4.25 bits/elem (packed nibbles + importance-weighted block scales).

All model hyperparameters read from GGUF metadata — nothing hardcoded:
`n_layers`, `hidden_dim`, `n_heads_q`, `n_heads_kv`, `ffn_intermediate`, `n_experts`, `n_active_experts`, `n_shared_experts`, `rope_theta`, `rope_type`, `max_seq_len`, `vocab_size`, `n_mtp_modules`, `mtp_depth`

---

## 7. Shader Dispatch Order

### Per layer (decode)

| Step | Shader | WG Layout | Barrier |
|---|---|---|---|
| RMSNorm | `rms_norm.comp` | (dim/256, 1, 1) | exec-only |
| QKV project | `qkv.comp` | (dim/64, n_heads_q, 1) | exec-only |
| RoPE | `rope.comp` | (n_heads_q, 1, 1) | exec-only |
| GQA attention | `attention.comp` | (n_heads_q, 1, 1) | exec-only |
| KV cache write | `kv_write.comp` | (n_kv_heads, 1, 1) | memory |
| Residual add | `residual_add.comp` | (dim/256, 1, 1) | exec-only |
| Router | `router_topk.comp` | (1, 1, 1) | exec-only |
| MoE gate+up | `moe_gate_up.comp` | (n_active, dim/64, 1) | exec-only |
| SiLU × gate | `silu_mult.comp` | (n_active, dim/64, 1) | exec-only |
| MoE down | `moe_down.comp` | (n_active, dim/64, 1) | exec-only |
| Combine experts | `moe_combine.comp` | (dim/256, 1, 1) | exec-only |
| Residual add | `residual_add.comp` | (dim/256, 1, 1) | exec-only |

### End of forward pass

| Step | Barrier |
|---|---|
| RMSNorm + LM Head | exec-only |
| Sample (GPU-side) | memory (CPU reads token) |
| MTP draft (blocks 0..depth-1) | exec-only chain |
| Verify pass (full model on drafts) | memory (CPU reads accepted count) |

---

## 8. Startup Flow

1. Parse GGUF header, validate model architecture matches Qwen 3.6 MoE
2. Allocate all buffers (weights, KV cache, compute intermediates) in single VkDeviceMemory
3. Compile all shaders from embedded SPIR-V (or GLSL→spirv at build time via `glslangValidator`)
4. Precompute dispatch parameters for each layer (push constant blocks, workgroup counts)
5. Pin server thread to CCX0
6. Start hyper server on port 8787

---

## 9. Chat Template + Stop Conditions

Qwen 3.6 chat format (Jinja2, parsed from `tokenizer_config.json`):
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
```

Stop conditions checked per SSE-frame:
- `<|im_end|>` token emitted → `stop_reason: "end_turn"`
- Stop sequence string match on accumulated text → `stop_reason: "stop_sequence"`
- Tool call JSON complete + model stops → `stop_reason: "tool_use"`
- `max_tokens` reached → `stop_reason: "max_tokens"`

## 10. Error Handling

| Error | Response |
|---|---|
| Invalid `model` field | 400 + `{"type":"error","error":{"type":"invalid_request_error","message":"..."}}` |
| Missing `messages` | 400 |
| `max_tokens` > context remaining | Clamp, warn in response metadata |
| Vulkan device lost | 500, log error, exit (no meaningful recovery on iGPU) |
| Out of KV cache | Return error, suggest reducing `max_tokens` or restart with shorter context |
| GGUF parse failure | Exit at startup with clear error (bad file, wrong arch, unsupported quant) |

## 11. Non-Goals (Explicitly Skipped)

- Multi-user / batching — single sequence, one request at a time
- OpenAI API format — Anthropic Messages only
- GPU switching / multi-GPU — 680M is the only target
- Training / fine-tuning
- Image / vision inputs
- Web UI
