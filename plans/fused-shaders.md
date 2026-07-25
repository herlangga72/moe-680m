# Fused Shaders — Plan

## Goal

Eliminate global memory round-trips between sequential per-layer ops. Move bottleneck from DDR5 bandwidth toward compute TFLOPS.

## Current pipeline per GQA layer

```
RMS norm → tmp (4 KB write)
Q GEMM   → tmp+4K (8 KB write), read RMS norm
K GEMM   → tmp+8K (1 KB write), read RMS norm
V GEMM   → tmp+8.7K (1 KB write), read RMS norm
RoPE     → tmp+4K in-place (9.5 KB read + write)
barrier
KV Write → cache
Attention → tmp (4 KB write)
attn_output GEMM → tmp+extra (4 KB write)
barrier
Residual Add → ho (4 KB write + 4 KB read)
Router → routing_logits (M * 256 f16)
```

Each `→` is a global memory round-trip. ~10 per GQA layer × 10 GQA layers = 100 trips.

## Fusions

### Fusion 1: rms_norm_qkv_rope (highest impact, ~3 dispatches eliminated)

Combine RMS norm + Q GEMM + K GEMM + V GEMM + RoPE into one shader.

**Phase 1 — Reduction (256 threads):**
- Load 2048 f16 hidden → LDS, compute sum-of-squares tree reduction
- Compute `rms = sqrt(sum/n + eps)`, broadcast via LDS
- Store normalized hidden in LDS (2048 f16 = 4 KB)

**Phase 2 — Q/K/V GEMM (tiled per threadgroup, 64×8 coverage):**
- Read normalized hidden from LDS
- Read IQ4_XS weights from global, dequant inline
- Three GEMM output regions: Q[4096], K[512], V[512]
- Each thread accumulates 2×2 tiles per K-step

**Phase 3 — RoPE (in-register):**
- Q[4096] and K[512] in registers after GEMM
- Apply cos/sin rotation per head
- Write Q_K_V_RoPE output (5120 f16 = 10 KB)

Savings vs current:
```
Before: RMS norm write + read (8 KB), Q/K/V write + RoPE read (28 KB) = 36 KB, 4 dispatches, 4 barriers
After:  1 fused write (10 KB), 1 dispatch, 1 barrier
```
Net: −26 KB global traffic, −3 dispatches per GQA layer × 10 = −30 dispatches per token.

### Fusion 2: attn_residual (smaller impact, ~2 dispatches eliminated)

Combine attention output GEMM + residual add.

After attention writes result to tmp, instead of separate output GEMM dispatch (read tmp, write out_off) then residual add (read hi + out_off, write ho):

**Fused shader:**
- Read attention output + residual input
- attn_output GEMM with IQ4_XS weights
- Add residual in-register
- Write final ho output

Savings:
```
Before: attn_output write + read (8 KB), residual reads hi + out_off (8 KB), 2 dispatches
After:  1 dispatch, 1 write (4 KB), 1 read hi (4 KB)
```
Net: −12 KB global traffic, −1 dispatch, −1 barrier per layer × 40 = −40 dispatches.

### Fusion 3: moe_w1w3_w2

Already fused (SiLU in W2Scatter). No further fusion viable — different weight matrices per expert, can't share between w1 and w2.

## Dispatch plan per fused kernel

### rms_norm_qkv_rope
```
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

shared float norm_in[2048];    // normalized hidden state
shared float red[256];          // reduction buffer
shared float qk_weights[64][33]; // Q tile (same access pattern as w1_w3_fused)

void main() {
    uint tid = gl_LocalInvocationIndex;
    
    // Phase 1: RMS norm
    float sum_sq = 0.0;
    for (int i = 0; i < 8; i++) {
        float v = f16_to_f32(read_input(tid * 8 + i));
        red[tid] = v * v;  // partial sum
        norm_in[tid * 8 + i] = v;
    }
    // Tree reduction
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        barrier();
    }
    float rms = sqrt(red[0] / K + 1e-6);
    // Normalize in-place
    for (int i = 0; i < 8; i++) norm_in[tid * 8 + i] /= rms;
    barrier();
    
    // Phase 2: Q GEMM (tiled)
    // dispatch_id = gl_WorkGroupID.y → which output block (0..Q_N/8)
    // Each threadgroup covers 1 M × 8 N
    float accum[2][2] = {{0,0},{0,0}};
    for (uint k = 0; k < K; k += 32) {
        // Load norm tile from LDS, weights from global IQ4_XS
        // 2×2 MAC loop
    }
    // Write Q output (computed before K/V GEMM in same shader)
    
    // Phase 3: K GEMM (same pattern, different weights)
    // Phase 4: V GEMM (same pattern)
    // Phase 5: RoPE on Q and K in registers
    // Write final packed QKVRoPE output
}
```

### attn_residual
```
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Thread reads attn_output element, does GEMM with weight row,
// adds residual, writes ho
void main() {
    // Same as current attention output GEMM but fused with residual add
}
```

## Build

- New shader files: `rms_norm_qkv_rope.comp`, `attn_residual.comp`
- New PipelineType entries in pipeline.rs
- Update inference.rs `record_and_submit_layer` to use fused pipelines
- Fallback paths for prefill (M > 1) where LDS doesn't fit

## When to fuse

| M | Fusion benefit | LDS fit |
|---|---|---|
| 1 (generation) | High — eliminate barriers, save global traffic | Yes — 4 KB norm + 64 KB attention output fits |
| >1 (prefill) | Low — input doesn't fit LDS, can't avoid global | No — M × 2048 f16 doesn't fit for M > 16 |

Generation path: use fused kernels. Prefill path: keep current multi-dispatch.

## Verification

- Compare generated tokens between fused and unfused — must match exactly
- `cargo check --release` after each shader
- Benchmark: `cargo build --release && time ./target/release/moe-680m --smoke`

## Skipped

- **DeltaNet fusion**: RMS norm + DeltaNet QKV possible, but DeltaNet Step is recurrent (state-dependent) and can't merge. Save ~1 trip per DeltaNet layer. Low priority — DeltaNet already faster than GQA.
- **Router fusion**: Already single dispatch + GPU topk.
- **MoE fusion**: W1W3Fused → SiLU → W2Scatter already fused. Can't merge across weight matrices.
