---
type: concept
title: Inference Pipeline
date: 2026-07-25
bundle: moe-680m-docs
concepts: [architecture, memory-layout]
---

# Inference Pipeline

## Startup Sequence

```ascii
┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────────┐
│ GGUF     │   │ Vulkan   │   │ Arena     │   │ Weights  │   │ Create       │
│ mmap     │→  │ Init     │→  │ Alloc     │→  │ Load     │→  │ Pipelines    │
│ + Parse  │   │ (UMA)    │   │ 20+ GB    │   │ IQ4_XS   │   │ (17-22)      │
└──────────┘   └──────────┘   └───────────┘   └──────────┘   └──────────────┘
                                                                    │
                                                                    ▼
                                                          ┌──────────────────┐
                                                          │ Bind Arena       │
                                                          │ Descriptor       │
                                                          │ Init Engine      │
                                                          └──────────────────┘
```

1. **GGUF Parse:** Memory-map the model file, parse header + metadata + tensor index. Extract configuration (layer count, hidden size, expert count, etc.)
2. **Vulkan Init:** Load Vulkan, enumerate devices, find UMA-capable device (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT), create logical device with required features (16-bit storage, float16, buffer device address)
3. **Arena Alloc:** Compute arena layout from model config. Single contiguous UMA allocation (20+ GB)
4. **Weight Load:** Copy all tensors from GGUF → arena. Convert F16 tensors to IQ4_XS format on load. `madvise(DONTNEED)` on GGUF pages after copy
5. **Pipeline Create:** Load all SPIR-V blobs, create compute pipelines with shared pipeline layout
6. **Bind Descriptor:** Bind entire arena buffer to descriptor set slot 0
7. **Init Engine:** Create command pool, command buffers, fence, sampling state

## Per-Token Generation (M=1 fused path)

```ascii
┌──────────────────────────────────────────────────────────────────┐
│ embed_token: IQ4_XS dequant → f16 at hidden_ping[pos]            │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                 ┌─────────────────────────┐
                 │  40 Layer Loop           │
                 │  (30 DN + 10 GQA)        │
                 │                          │
                 │  ┌───────────────────┐   │
                 │  │ GQA (fused):      │   │
                 │  │ RMSNorm+QKV+RoPE  │───│──→ QKV buffer
                 │  │   → KV Write      │───│──→ KV cache
                 │  │   → Attention     │───│──→ attn output
                 │  │   → Attn+Residual │───│──→ ho
                 │  └───────────────────┘   │
                 │         OR               │
                 │  ┌───────────────────┐   │
                 │  │ DeltaNet (fused):  │   │
                 │  │ RMSNorm+QKV        │───│──→ QKV buffer
                 │  │   → Step (recurrent)│──│──→ state update
                 │  │   → Output+Residual│──│──→ ho
                 │  └───────────────────┘   │
                 │                          │
                 │  → Router GEMM           │───→ routing logits
                 │  → GPU TopK (if M=1)     │───→ expert indices
                 └─────────────────────────┘
                               │
                               ▼
                 ┌─────────────────────────┐
                 │  MoE: Expert FFN         │
                 │  9 experts (shared+8)    │
                 │  W1W3Fused → W2Scatter   │
                 │  Weighted scatter-add     │
                 └─────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     ┌──────────────────┐          ┌──────────────────────┐
     │ LM Head GEMM      │          │ MTP Heads (×2)        │
     │ hidden×emb → logits│         │ hidden×W1→SiLU×W2     │
     │ 248K f16→f32 SIMD │          │ → draft logits        │
     └────────┬─────────┘          └──────────┬───────────┘
              ▼                                ▼
     ┌──────────────────┐          ┌──────────────────────┐
     │ Sample token      │          │ Read drafts (argmax)  │
     │ (CDF binary search)│         │                      │
     └────────┬─────────┘          └──────────┬───────────┘
              └──────────┬──────────────────┘
                         ▼
              ┌──────────────────────┐
              │ Embed accepted tokens │
              │ (main + drafts)       │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ Verification Pass     │
              │ (run all layers on    │
              │  accepted batch)      │
              │ → extends KV cache    │
              └──────────────────────┘
```

### Generation (autoregressive loop)

1. **Embed:** Dequantize IQ4_XS embedding weight for token ID → f16 hidden state
2. **40 layers:** Each layer processes hidden state through RMS norm → QKV → attention/step → output projection → residual → router → MoE
3. **LM Head:** hidden × embedding^T → logits (248K vocab f16 → f32 via SIMD)
4. **Sample:** CDF binary search for temperature/top-k sampling (or argmax for greedy)
5. **MTP Drafts:** Read MTP head logits from GPU, greedy argmax per head
6. **Verify:** Embed all accepted tokens (1 main + N drafts), run one forward pass to extend KV cache
7. **Return** all accepted tokens

### Dispatch Comparison

| Layer path | Old (unfused) | Fused | Saved |
|---|---|---|---|
| GQA × 10 | 10 dispatches | 4 dispatches | 60 total |
| DeltaNet × 30 | 6 dispatches | 4 dispatches | 60 total |
| RouterTopk | 1 | 1 | 0 |
| **Total** | **261** | **141** | **120** |
