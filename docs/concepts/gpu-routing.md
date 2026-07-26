---
type: concept
title: GPU Routing for Prefill
date: 2026-07-25
bundle: moe-680m-docs
concepts: [architecture, inference-flow]
---

# GPU Routing for Prefill

## Problem

Prefill routing was CPU-only: read M×256 f16 logits from coherent memory → convert to f32 → softmax → top-8 → prefix-sum buckets. 40× per prefill. Each token processed by `route_single()` — softmax + full sort.

## Solution

The `router_topk` shader already supported M>1 (via `gl_GlobalInvocationID.x` per-token dispatch), but the Rust code only enabled it for M=1. Removed the guard, extended dispatch to `(M, 1, 1)`.

## Before vs After

| Aspect | Before | After |
|---|---|---|
| M=1 routing | GPU softmax + topk, read 72 bytes | Same (no change) |
| M>1 routing | CPU: f16 readback → f32 → softmax → sort | GPU: softmax + topk, read M×72 bytes |
| CPU work per layer | `M × (softmax + sort_256)` | `M × unpack_8_entries` |
| Memory read per layer | `M × 256 × 2` bytes f16 | `M × 72` bytes compact |
| CPU softmax calls | 40 × M | 0 |
| `route_single()` calls | 40 × M | 0 |

## Data Format

router_topk writes `M × 9 × 8` bytes:
```
entry 0:   shared expert (id=0, weight=1.0)
entry 1-8: top-8 routed experts, sorted descending
layout:    u32:expert_id_low16 | u32:weight_f32
```

## Dispatch

```rust
// Previously: only for M==1
// Now: for all M
self.bind_pipe(cmd, PipelineType::RouterTopk);
pc.M = M;
self.push(cmd, &pc);
unsafe { dev.cmd_dispatch(cmd, M.max(1), 1, 1); }
```

## Resource Usage

Routing topk buffer resized from 256 bytes to `min(8192, ctx_len) × 72` bytes (max ~576 KB). Capped at same prefill batch size as temp region.
