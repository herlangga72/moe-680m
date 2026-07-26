---
type: concept
title: Rust Modules
date: 2026-07-25
bundle: moe-680m-docs
concepts: [architecture, inference-flow]
---

# Rust Modules

11 source files, ~3,200 total lines. Architecture: single-threaded, synchronous GPU dispatch.

## Module Dependency Map

```ascii
  ┌──────────┐     ┌──────────┐     ┌──────────────┐
  │  main.rs  │────→│ tui.rs   │     │  server.rs   │
  │ (CLI/TUI) │     │ (config) │     │  (HTTP API)  │
  └─────┬────┘     └──────────┘     └──────┬───────┘
        │                                  │
        ▼                                  │
  ┌──────────┐                             │
  │ gguf.rs  │  (model parsing)            │
  └────┬─────┘                             │
       │                                   │
       ▼                                   │
  ┌──────────┐     ┌──────────┐     ┌──────┴───────┐
  │ memory.rs│────→│ device.rs│     │ inference.rs │
  │ (arena)  │     │ (Vulkan) │     │ (engine)     │
  └──────────┘     └──────────┘     └───┬──┬───┬───┘
       │                                │  │   │
       ▼                                ▼  │   ▼
  ┌──────────┐     ┌──────────┐     ┌──────┴───────┐
  │ util.rs  │     │pipeline.rs│    │  router.rs   │
  │ (f16,     │     │(shaders)  │    │  (MoE route) │
  │  SIMD)   │     └──────────┘    └──────────────┘
  └──────────┘
                              ┌──────────────┐
                              │ sampling.rs  │
                              │ (token pick) │
                              └──────────────┘
                              ┌──────────────┐
                              │ tokenizer.rs │
                              │ (BPE encode) │
                              └──────────────┘
```

## File Roles

### `main.rs` — Entry Point (381 lines)

- CLI argument parsing (manual, no clap)
- TUI launch (default with no args)
- `run_smoke()` — Vulkan device enumeration + UMA test
- `run_inference()` — model load → engine init → generation loop
- `chat_loop()` — multi-turn conversation with incremental KV cache

### `device.rs` — Vulkan Context (300 lines)

- `DeviceContext` struct: device, queue, memory type, capabilities
- VMA detection: finds memory type with `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`
- Feature chain: 16-bit storage → float16 → buffer device address
- UMA allocation: `allocate_uma(size)` + `map_memory`
- Buffer creation: `create_buffer_from_memory(memory, size)`

### `memory.rs` — Memory Management (369 lines)

- `ArenaLayout::compute()` — calculates all region offsets from `ModelConfig`
- `TensorRegistry` — name→offset HashMap, sorted load order
- `LayerWeights` — per-layer offset arrays for all weight types
- `load_weights_from_tensors()` — copies GGUF tensors to arena, F16→IQ4_XS conversion
- `quantize_f16_to_iq4xs()` — on-load quantization for embedding table

### `pipeline.rs` — Compute Pipeline (259 lines)

- `MAX_PIPELINES = 24` — slot for each shader type
- `PipelineType` enum — 21 variants (17 active + 4 fused)
- `create_pipelines()` — descriptor set, pool, layout, pipeline cache
- `bind_arena_descriptor()` — binds whole arena to binding 0

### `inference.rs` — Inference Engine (884 lines)

- `PushConstants` — 128B struct, bytemuck Pod/Zeroable
- `InferenceEngine` — device, pipelines, arena, command buffers, fence
- `embed_token()` — IQ4_XS dequant → f16 (SIMD batch f32→f16)
- `prefill()` / `prefill_incremental()` — prompt processing
- `forward_single()` — single token generation, lm_head, sampling
- `generate_mtp()` — MTP speculative decoding with verification
- `record_and_submit_layer()` — per-layer command buffer (fused or unfused)
- `record_and_submit_moe()` — per-layer MoE dispatch
- `do_route()` — GPU topk readback or CPU prefill routing

### `router.rs` — MoE Routing (98 lines)

- `RoutingOutput` — 8 expert indices + weights per token
- `route_single()` — softmax + `select_nth_unstable_by` for top-8
- `build_expert_batches()` — prefix-sum bucket fill for prefill
- `route_cpu()` — (dead code, kept as reference)

### `sampling.rs` — Token Sampling (103 lines)

- `SamplingParams` — temperature, top_k, max_tokens
- `SamplingContext` — rng_state, past_tokens, token frequencies
- `sample()` — temperature scale → softmax → CDF → branchless binary search
- `xorshift_f32()` — xorshift PRNG

### `tokenizer.rs` — BPE Tokenizer (153 lines)

- `TokenizerData` — from GGUF metadata (tokens, scores, merges)
- `Tokenizer` — BPE encode/decode, BOS/EOS handling
- Optional `gigatoken` feature for exact tiktoken BPE

### `server.rs` — HTTP API (173 lines, feature-gated)

- `POST /v1/messages` — Anthropic Messages API format
- JSON + SSE (streaming) responses
- Single-user, synchronous (Mutex-wrapped engine)

### `tui.rs` — Interactive TUI (291 lines)

- Zero-dependency terminal UI via `libc` raw mode
- Arrow key navigation, inline editing, field validation
- Chat mode toggle, run/smoke/quit actions

### `util.rs` — Shared Utilities (109 lines)

- `VOCAB_SIZE = 248320`
- `f16_bits_to_f32` / `f32_to_f16_bits` — branchless arithmetic
- `f16_slice_to_f32` — SIMD `_mm256_cvtph_ps` + scalar fallback
- `f32_slice_to_f16` — SIMD `_mm256_cvtps_ph` + scalar fallback
- `argmax` — manual loop (LLVM auto-vectorizes)
