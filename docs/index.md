---
type: bundle
title: moe-680m Inference Engine
version: 0.1.0
date: 2026-07-25
architecture: knowledge-base
tags: [moe, inference, vulkan, rdna2, llm]
bundle: moe-680m-docs
---

# moe-680m Inference Engine

MoE (Mixture of Experts) inference engine for **AMD Radeon 680M** (RDNA2 APU). Runs Qwen3.6-35B-A3B at ~15-20 tok/s on shared DDR5 memory via Vulkan Compute.

## Quick Links

- [Architecture](concepts/architecture.md) — system design, layers, experts
- [Inference Flow](concepts/inference-flow.md) — prefill, generation, MTP decode
- [Memory Layout](concepts/memory-layout.md) — arena allocator, tensor registry, weight loading
- [Shaders](shaders/index.md) — all 22 GLSL compute shaders
- [Rust Modules](rust/index.md) — all source files and their roles
- [Flows](flows/index.md) — generation, prefill, chat flows
- [Build System](concepts/build-system.md) — Makefile, Cargo, features

## Repository Structure

```
moe-680m/
├── src/               # Rust source (11 files)
│   ├── main.rs        # CLI entry, TUI wiring, chat loop
│   ├── device.rs      # Vulkan init, UMA detection
│   ├── memory.rs      # Arena layout, tensor registry, weight loading
│   ├── pipeline.rs    # SPIR-V → compute pipelines
│   ├── inference.rs   # Inference engine (fused + unfused paths)
│   ├── router.rs      # MoE routing (GPU topk + CPU batching)
│   ├── sampling.rs    # Token sampling (CDF binary search)
│   ├── tokenizer.rs   # BPE tokenizer
│   ├── server.rs      # Anthropic-compatible HTTP API
│   ├── tui.rs         # Interactive terminal UI
│   └── util.rs        # f16↔f32, SIMD batch, argmax
├── shaders/           # GLSL compute shaders (22 files)
│   ├── common.glsl    # Shared helpers: f16, dequant, SiLU
│   ├── w1_w3_fused.comp, w2.comp, w2_scatter.comp  # MoE GEMM
│   ├── attention.comp, rope.comp, kv_write.comp     # Attention
│   ├── deltanet_*.comp   # DeltaNet shaders
│   └── rms_norm_qkv_rope.comp, attn_residual.comp   # Fused
├── docs/              # OKF knowledge bundle
│   ├── index.md       # ← you are here
│   ├── concepts/      # Architecture, memory, build
│   ├── shaders/       # Shader documentation
│   ├── rust/          # Rust module docs
│   └── flows/         # Process flows
└── plans/             # Design docs, bug log
```

## Key Numbers

| Metric | Value |
|---|---|
| Total params | 35B |
| Active per token | 3B |
| Experts | 256 routed + 1 shared |
| Routed per token | 8 |
| Layers | 40 (30 DeltaNet + 10 GQA) |
| Hidden size | 2048 |
| Intermediate (FFN) | 4128 |
| Context length | 262K |
| Vocab | 248,320 |
| Weight quant | IQ4_XS (4.5 bpw) |
| KV cache quant | Q4_0 (4.5 bpw) |
| Weight size | ~18.2 GB |
| VRAM (UMA shared) | ~20 GB |
| Generation speed | 15-20 tok/s |
| First token (512 prefill) | ~2s |

## Hardware Target

- **GPU:** AMD Radeon 680M (RDNA2, 12 CUs, 2.2 GHz)
- **CPU:** Ryzen 7 6800H (Zen 3+, 8 cores)
- **Memory:** Dual-channel DDR5-4800 (~76 GB/s shared UMA)
- **Peak compute:** 3.38 TFLOPS (f32), 6.76 TFLOPS (f16 packed)
- **Bottleneck:** DDR5 bandwidth (GEMM utilization ~0.9% of peak)
