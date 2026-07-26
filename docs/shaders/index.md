---
type: concept
title: Shaders
date: 2026-07-25
bundle: moe-680m-docs
concepts: [architecture]
---

# GLSL Compute Shaders

22 shaders total (18 active + 4 stub/unused). All target Vulkan 4.50 with `scalar_block_layout`. Shared `common.glsl` included by all.

## Shared Infrastructure (`common.glsl`)

| Function | Purpose | RDNA2 Target |
|---|---|---|
| `read_u8/16/32` | Byte/dword reads from arena via `data[off>>2]` with shift+mask | Scalar load |
| `f16_to_f32` | Half→single via `unpackHalf2x16(bits).x` | `v_cvt_f32_f16` (1 instr) |
| `f32_to_f16` | Single→half via `packHalf2x16(vec2(v,0))` | `v_cvt_pkrtz_f32_f16` (1 instr) |
| `dequant_iq4_xs` | IQ4_XS block → f32 with d0/d2 selection | Inline |
| `read_kv_q4_0` | Q4_0 cache → f32 | Inline |
| `silu` | SiLU activation `x/(1+exp(-x))` | Inline |

## GEMM Kernels (wave32, 32×4 threadgroups)

| Shader | Input | Weight | Output | Coverage |
|---|---|---|---|---|
| `w1_w3_fused` | f16 hidden | IQ4_XS MoE gate+up | f16 scratch | 64 M × 8 N |
| `w2.comp` | f16 scratch | IQ4_XS MoE down | f16 hidden | 64 M × 8 N |
| `w2_scatter` | f16 gate||up fused | IQ4_XS+SiLU | f16 hidden+scatter | 64 M × 8 N |
| `attn_output` | f16 attn result | IQ4_XS output proj | f16 buffer | 64 M × 8 N |
| `qkv.comp` (=attn_output) | f16 normed | IQ4_XS Q/K/V proj | f16 buffer | 64 M × 8 N |

LDS: `At[64][33]` (8 KB, padded 33→32 avoids bank conflict) + `Bt[32][8]` (1 KB). TILE_K=32.

## Attention

| Shader | Dispatch | Purpose |
|---|---|---|
| `rope.comp` | (M, 18, 1) × 64 threads | RoPE on Q (16 heads) + K (2 heads), first 64 dims |
| `kv_write.comp` | (div64(M), 32, 1) × 32 threads | Q4_0 quantize K/V, write to cache |
| `attention.comp` | (M, 16, 1) × 256 threads | Flash attention with online softmax, Q4_0 inline dequant |

## DeltaNet

| Shader | Dispatch | Purpose |
|---|---|---|
| `deltanet_qkv.comp` | (div64(M), div8(4128), 1) × 128T | QKV projection (4128 f16 output) |
| `deltanet_step.comp` | (16, 32, 1) × 128T | Recurrent state update: S = g·S + (1-g)·outer(k,v) |
| `deltanet_output.comp` | (div64(M), div8(2048), 1) × 128T | Output projection: 4096→2048 |

## Utility Kernels

| Shader | Dispatch | Purpose |
|---|---|---|
| `rms_norm.comp` | (M, 1, 1) × 256T | RMS layer norm, tree sum-sq reduction |
| `silu_mult.comp` | (M, div256(N), 1) × 256T | Element-wise SiLU activation |
| `residual_add.comp` | (M, 1, 1) × 256T | Element-wise hidden + gemm → output |
| `router.comp` | (div64(M), div8(256), 1) × 128T | Router GEMM: hidden × weight → 256 logits |
| `router_topk.comp` | (1, 1, 1) × 256T | Softmax + bitonic top-8 on GPU |

## Fused Generation Shaders (M=1)

| Shader | Replaces | Saves |
|---|---|---|
| `rms_norm_qkv_rope.comp` | RMS norm + Q GEMM + K GEMM + V GEMM + RoPE | 4 writes + 3 reads, 3 barriers |
| `attn_residual.comp` | AttnOutput GEMM + ResidualAdd | 1 write + 2 reads, 1 barrier |
| `rms_norm_dn_qkv.comp` | RMS norm + DeltaNet QKV | 1 write + 1 read, 1 barrier |
| `dn_output_residual.comp` | DeltaNet output + ResidualAdd | 1 write + 2 reads, 1 barrier |

## Unused / Stub

| Shader | Status |
|---|---|
| `w1_w3_fused_m1.comp` | M1 prefill variant, compiled but never loaded |
| `w2_scatter_m1.comp` | M1 prefill variant, compiled but never loaded |
| `moe_combine.comp` | Compiled but unused (scatter-add inline in W2Scatter) |

## Build

```
make all   # glslc → SPIR-V, auto-detects all .comp files
make gemm  # fast iteration: only GEMM kernels
```

SPIR-V output goes to `src/shaders/*.spv`, embedded via `include_bytes!()` in `pipeline.rs`.
