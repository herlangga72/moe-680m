---
type: log
title: Changelog
date: 2026-07-25
bundle: moe-680m-docs
---

# Changelog

## 2026-07-25 — fused-shaders branch

### Added
- Fused GQA generation shader `rms_norm_qkv_rope`: RMS norm + QKV GEMM + RoPE in one dispatch
- Fused attention output + residual shader `attn_residual`
- Fused DeltaNet generation shader `rms_norm_dn_qkv`: RMS norm + DN QKV
- Fused DeltaNet output + residual shader `dn_output_residual`
- Interactive TUI with arrow-key navigation (`src/tui.rs`)
- Interactive chat mode (`--chat` CLI flag)
- SKILL.md following OKF skill template

### Fixed
- **GQA K/V output offsets:** K/V projection offsets were `+4096` / `+4608` bytes (element offset). Should be `+8192` / `+9216` (element × f16 size). kv_write shader expects contiguous 5120-element stride per token.
- **TUI editing in raw mode:** `edit_field` ran with ICANON/ECHO off, no visible input
- **Escape sequence desync:** Non-arrow function keys left garbage bytes on stdin
- **Chat BOS missing:** Multi-turn conversation encoded without BOS, state seq_len mismatch
- **arena_buffer leak:** VkBuffer handle never destroyed
- **DeltaNet state double-offset:** Both Rust and shader added `layer × LAYER_SIZE` to state base (fixed in earlier refactor)

## 2026-07-24 — master branch

### Added
- RDNA2 native f16 conversions (`unpackHalf2x16` / `packHalf2x16`) in all shaders
- SIMD batch f16→f32 (`_mm256_cvtph_ps`) and f32→f16 (`_mm256_cvtps_ph`)
- Branchless CDF binary search for sampling (removes unpredictable `if r < cum` linear scan)
- Branchless f16↔f32 conversions on CPU via arithmetic masking
- Reusable logits buffer (eliminates 992 KB heap alloc per token)
- Pre-allocated routing token/weight regions (separated from GPU scratch)

### Performance
- ~120 dispatches eliminated per generation token (down from ~261)
- ~740 KB global traffic saved per token via fused shaders
- Full dispatch counts: GQA 10→4, DeltaNet 6→4 per layer
