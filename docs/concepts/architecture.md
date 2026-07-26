---
type: concept
title: System Architecture
date: 2026-07-25
bundle: moe-680m-docs
concepts: [inference-flow, memory-layout, shaders, build-system]
---

# Architecture

## Overview

moe-680m is a Vulkan Compute inference engine for MoE (Mixture of Experts) transformer models on AMD RDNA2 APUs. Single UMA memory allocation, inline-dequant GEMM, on-the-fly command buffer recording.

```ascii
                         ┌──────────────────────────┐
                         │     Vulkan Compute        │
                         │   (ash 0.38, RADV)        │
                         │   1 descriptor set         │
                         │   128B push constants      │
                         └───────────┬──────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
 ┌──────────────┐          ┌──────────────────┐        ┌──────────────────┐
 │   40 Layers   │          │   256 Experts     │        │  IQ4_XS + Q4_0   │
 │ hybrid        │          │   8 routed        │        │  18.2 GB weights  │
 │ DN + GQA      │          │   +1 shared       │        │  1.5 GB KV cache  │
 └──────┬───────┘          └──────────────────┘        └──────────────────┘
        │
        ├── 30× DeltaNet (0,1,2,4,5,6,8,9,10,...)
        │    Linear attention, recurrent state (16 QK × 32 V heads × 128²)
        │
        └── 10× GQA (3,7,11,...,39)
             Standard attention, Q4_0 KV cache (16 Q × 2 KV heads × 256 dim)

```

## Hybrid Architecture

The model mixes two attention mechanisms across 40 layers. GQA layers occur every 4th layer (indices 3, 7, 11, ..., 39). All other layers are DeltaNet.

### GQA (Grouped Query Attention)

Standard multi-head attention with GQA 8:1 ratio (16 Q heads, 2 KV heads). KV cache stored in Q4_0 format (4.5 bpw). RoPE applied to first 64 dims of each head.

**Layer pipeline (M=1 fused):**
```
RMSNorm+QKV+RoPE (fused) → KV Write → Attention → AttnOutput+Residual (fused) → Router
```

### DeltaNet (Gated Linear Attention)

Linear attention with recurrent state update. State per (QK head × V head) pair: 128×128 f32 matrix. State persists across tokens — no KV cache needed.

**State update:** `S = gate·S + (1−gate)·outer(k, v)`, where `k,q,v,gate` are learned projections of the normalized input.

**Output:** `o = S × q` — state-vector product per head pair, concatenated across V heads.

## MoE (Mixture of Experts)

- **256 routed experts** + 1 shared expert per layer
- **8 experts selected per token** via softmax routing
- **Shared expert always active** — provides stable gradient backbone
- **IQ4_XS weights** — 4.5 bpw importance-aware quantization

### Routing

| Mode | Method | Batch | Latency |
|---|---|---|---|
| Prefill (M>1) | CPU: f16 logits readback → softmax → top-8 → prefix-sum bucket fill | M×9 slots, per-expert dispatches | ~5μs/token/layer |
| Generation (M=1) | GPU: router_topk shader → softmax + top-8 → CPU reads 9×8 bytes | 1 token, sequential per expert | ~1μs |

## Memory Architecture

Single UMA allocation (`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`). All weights, states, cache, and scratch in one contiguous buffer. CPU and GPU share the same physical DDR5 memory — no copies.

[See full memory layout →](memory-layout.md)

## Compute Pipeline

- **Single descriptor set** (binding 0): 1 storage buffer covering the entire arena
- **128-byte push constants** shared by all pipelines: offsets, dims, parameters
- **17+ pipeline types** indexed by enum, loaded from embedded SPIR-V (`include_bytes!`)
- **On-the-fly command buffer recording:** ~5μs overhead per layer, no pre-recorded buffers
- **Pipeline cache:** auto-saved to `$TMPDIR/moe_pipeline_cache.bin`

[See shaders →](../shaders/index.md)
[See Rust modules →](../rust/index.md)
