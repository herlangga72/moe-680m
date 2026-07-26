---
type: concept
title: Memory Layout
date: 2026-07-25
bundle: moe-680m-docs
concepts: [architecture]
---

# Memory Layout

## Arena System

Single contiguous UMA allocation (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`). All model data lives in one buffer. CPU and GPU access the same physical DDR5 — zero copies.

Layout computed by `ArenaLayout::compute()` in `memory.rs`. All offsets aligned to 64 bytes. Offset 0 is reserved as null sentinel (weight lookup returns 0 for missing tensors).

```ascii
┌──────────────────────────────────────────────────────────────────┐
│ 0                Arena Layout (not to scale)                      │
├──────────────────────────────────────────────────────────────────┤
│ [null sentinel]  ~18.2 GB                                        │
│                  ┌────────────────────────────────────────────┐   │
│ weights_base ───→│ IQ4_XS weight matrices (40 layers × 256    │   │
│                  │  experts × 3 weight types + dense weights) │   │
│                  └────────────────────────────────────────────┘   │
│ hidden_ping ────→│ f16 ping buffer (ctx_len × 2048 × 2B)       │
│ hidden_pong ────→│ f16 pong buffer (same size)                  │
│ kv_cache_base ──→│ Q4_0 KV cache (10 GQA layers)                │
│                  │  per layer: ctx_len × 576 B                   │
│                  │  kv_cache_layer_stride = per-layer slice       │
│ deltanet_state → │ f32 state (30 layers × 16×32×128×128×4B)    │
│ scratch_base ───→│ MoE intermediate compute (GPU-only)          │
│ temp_base ──────→│ Layer compute temp (QKV, attention output)   │
│                  │  capped at 8192 tokens                        │
│ routing_logits → │ f16 routing logits (ctx_len × 256)           │
│ routing_topk ───→│ GPU topk results (9 × 8B = 72B used, 256B   │
│                  │  allocated)                                   │
│ routing_token ──→│ CPU→GPU: sorted token IDs for MoE batches    │
│ routing_weight → │ CPU→GPU: sorted routing weights for MoE      │
└──────────────────────────────────────────────────────────────────┘
```

## Tensor Registry

Name → offset mapping built at load time. Tensors laid out in a specific order for cache efficiency:

1. `token_embd.weight` — embedding table (accessed every token)
2. Per-layer dense: `attn_q, attn_k, attn_v, attn_output, ffn_gate, shared_expert.w1/w2/w3`
3. Expert weights: `blk.{L}.experts.{E}.w1/w2/w3.weight` for all 40 layers × 256 experts
4. Remaining tensors (output norm, MTP heads, etc.) appended in file order

## Weight Format: IQ4_XS

Importance-aware 4-bit quantization. 36 bytes per 32-weight block.

```
Block layout (36 bytes):
  [d0: f16] [d2: f16] [nibbles: 16B] [qh: 16B]
   d0 = scale for weights 0-15
   d2 = scale for weights 16-31
   nibbles[i] = 4-bit quant for each weight (2 per byte)
   qh[i] = 5th bit (high bit) for each weight
```

Dequant: `value = (nibble + high_bit×16 - 8) × scale`

## KV Cache: Q4_0

18 bytes per 32-weight block. Used for K and V in GQA attention layers.

```
Q4_0 block (18 bytes):
  [d: f16] [nibbles: 16B]
  d = scale
  nibbles[i] = 4-bit quant for each weight
```

Per token per GQA layer: K=288B + V=288B = 576B total.

## Communication Regions

Arena regions annotated with data flow direction:

| Region | Direction | Content |
|---|---|---|
| routing_logits_base | GPU→CPU | Router logits (f16) for prefill routing |
| routing_topk_base | GPU→CPU | Top-8 expert IDs + weights (72B) for generation |
| routing_token_base | CPU→GPU | Sorted token IDs for MoE expert batches |
| routing_weight_base | CPU→GPU | Sorted routing weights for MoE |

## DeltaNet State

Recurrent state: 128×128 f32 per (QK head, V head) pair. 16 QK × 32 V = 512 pairs per layer. 30 layers = 15,360 states × 16,384 B each ≈ 960 MB.
