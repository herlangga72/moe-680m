---
type: concept
title: Fused Generation Shaders
date: 2026-07-25
bundle: moe-680m-docs
concepts: [architecture, inference-flow, shaders]
---

# Fused Generation Shaders

Fused kernels combine sequential operations into single dispatches to eliminate global memory round-trips and barrier overhead. Active only for M=1 (generation). Prefill (M>1) uses original multi-dispatch.

## GQA Fusion: `rms_norm_qkv_rope`

Replaces: RMS Norm → Q GEMM → K GEMM → V GEMM → RoPE (5 dispatches → 1)

```
Before:            After:
  hi ─→ RMS ─→ tmp      hi ───┐
       Q GEMM ─→ tmp+H         │
       K GEMM ─→ tmp+H+K       ├── Fused ─→ out_off (QKVRoPE)
       V GEMM ─→ tmp+H+K+V     │   1 dispatch, 640 TGs
       RoPE ─→ (in-place)  ───┘
```

### Kernel Structure

```ascii
Phase 1: RMS Norm (128 threads)
  ─ load 2048 f16 from input → sum_sq → tree reduce → norm_in[2048] in LDS

Phase 2: GEMM (per threadgroup, 8 columns)
  ─ determine region: Q (oc<4096) | K (4096≤oc<4608) | V (oc≥4608)
  ─ read norm_in from LDS (no global read)
  ─ dequant IQ4_XS weights per K-tile
  ─ 2×2 MAC accumulation

Phase 3: RoPE + Write
  ─ if Q/K element in first 64 dims: apply rotation, write pair
  ─ if V element: write as-is
```

### Push Constants

| Rust field | GLSL field | Use |
|---|---|---|
| `input_offset` | `input_offset` | Hidden state (hi) |
| `weights_offset` | `weights_offset` | Q weight matrix |
| `output_offset` | `output_offset` | QKV output (out_off) |
| `M` | `M` | 1 |
| `N` | `N` | 5120 (total Q+K+V) |
| `K` | `K` | 2048 (hidden size) |
| `num_experts` | `position` | RoPE position (seq_len) |
| `routing_weights_off` | `k_weights_off` | K weight matrix |
| `layer_idx` | `layer_idx` | Layer index |
| `token_ids_off` | `v_weights_off` | V weight matrix |

### LDS Budget

| Buffer | Size | Purpose |
|---|---|---|
| `norm_in[2048]` | 8 KB | Normalized hidden state (f32) |
| `red[128]` | 0.5 KB | RMS norm reduction |
| `Bt[32][8]` | 1 KB | GEMM weight tile |
| **Total** | **9.5 KB** | (64 KB available) |

## GQA Fusion: `attn_residual`

Replaces: AttnOutput GEMM → ResidualAdd (2 dispatches → 1)

Same GEMM tiling as `attn_output.comp` but adds residual input inline. Reads attention output from `tmp`, dequantizes `attn_output` weight (IQ4_XS), writes result + hi → ho.

## DeltaNet Fusion: `rms_norm_dn_qkv`

Replaces: RMS Norm → DeltaNet QKV GEMM (2 dispatches → 1)

Same structure as GQA norm+GEMM but outputs 4128 elements (DN QKV format: Q2048 + K2048 + V32 + gate32).

## DeltaNet Fusion: `dn_output_residual`

Replaces: DeltaNet Output GEMM → ResidualAdd (2 dispatches → 1)

Reads QKV buffer at stride 4128. GEMM from first 4096 elements → 2048 hidden. Adds hi residual. Writes ho.
