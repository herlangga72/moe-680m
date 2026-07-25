# moe-680m

MoE inference engine for **AMD Radeon 680M** (RDNA2, APU). Runs Qwen3.6-35B-A3B at ~15-20 tok/s on shared DDR5 memory.

## Architecture

```
                   ┌──────────────────────┐
                   │   Vulkan Compute      │
                   │   (ash 0.38, RADV)    │
                   └──────────┬───────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐       ┌──────────────┐     ┌──────────────┐
   │ 40 Layers │       │  256 Experts  │     │  Q4_0 KV     │
   │ hybrid    │       │  8 routed     │     │  Cache       │
   │ DN+GQA    │       │  +1 shared    │     │  1.51 GB     │
   └──────────┘       └──────────────┘     └──────────────┘
```

- **35B total params, 3B active per token** — MoE with 256 experts (8 routed + 1 shared)
- **Hybrid architecture** — 30 Gated DeltaNet layers (linear attention) + 10 GQA layers
- **UMA memory** — single `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` allocation
- **IQ4_XS quantization** — all weights in importance-aware 4-bit format (~18.2 GB)
- **Q4_0 KV cache** — 4.5 bpw, 1.51 GB for 262K context (vs 5.36 GB f16)
- **On-the-fly command buffer recording** — no pre-recorded buffers, ~5μs overhead per layer

## Requirements

- **GPU:** AMD Radeon 680M (RDNA2, 12 CUs) or compatible Vulkan 1.3 device with UMA
- **Driver:** Mesa RADV (amdgpu) — provides UMA memory type exposure
- **RAM:** 27 GB+ available (model: 18.2 GB + KV cache: 1.5 GB + scratch: ~1 GB)
- **Vulkan SDK:** `glslc` for shader compilation

## Quick Start

```bash
# Build
git clone git@github.com:herlangga72/moe-680m.git
cd moe-680m
make all                    # compile GLSL shaders → SPIR-V
cargo build --release       # build Rust binary

# Download a Qwen3.6 GGUF model
# e.g. from https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF

# Run inference
./target/release/moe-680m \
    --model qwen3.6-35b-a3b-UD-IQ4_XS.gguf \
    --prompt "Hello, how are you?" \
    --max-tokens 200

# Smoke test (Vulkan device enumeration)
./target/release/moe-680m --smoke
```

## CLI Options

```
--model <PATH>       Load GGUF model
--prompt <TEXT>      Text prompt
--max-tokens <N>     Max tokens to generate (default 100)
--server [PORT]      Start HTTP server (Anthropic API, default 8080)
--smoke              Vulkan smoke test (device enumeration + UMA check)
--debug              Enable debug logging (or MOE_DEBUG=1)
--validate           Enable Vulkan validation layers (MOE_VK_VALIDATION=1)
```

## Performance

| Model | Quant | Context | Est. tok/s | Bottleneck |
|-------|-------|---------|------------|------------|
| Qwen3.6-35B-A3B | IQ4_XS | 262K | 15-20 | DDR5 bandwidth |
| Prefill (512 tok) | — | — | ~2s first token | Compute (12 CUs) |

## Project Structure

```
src/
├── main.rs         — CLI entry point, inference pipeline wiring
├── device.rs       — Vulkan init, UMA detection, buffer creation
├── pipeline.rs     — SPIR-V → compute pipelines, descriptor binding
├── memory.rs       — Arena layout, tensor registry, weight loading
├── gguf.rs         — GGUF parser (header, metadata, tensor index)
├── inference.rs    — 40-layer loop, MoE dispatch, prefill batching
├── router.rs       — CPU 256-way softmax, top-8, expert assignment
├── sampling.rs     — Temperature, top-k/p, penalties, stop conditions
├── tokenizer.rs    — BPE tokenizer from GGUF vocab (Gigatoken optional)
└── server.rs       — Anthropic-compatible HTTP API (--features server)

shaders/            — 15 GLSL compute shaders
plans/              — Design docs, gap analysis, implementation plans
```

## Key Design Decisions

**RDNA2 wave32:** All GEMM kernels use 32×4 threadgroups (128 threads, 4 waves) with 2×2 per-thread tiles = 64 M × 8 N coverage per threadgroup. TILE_K=32 halves barrier count vs 16.

**Branchless conversions:** `f16_to_f32` and `f32_to_f16` use arithmetic instead of conditional branches. Denormals flushed to zero (acceptable for quantized inference).

**Q4_0 KV cache:** K/V stored as Q4_0 blocks (18 bytes per 32 weights = 4.5 bpw). Attention shader dequantizes inline. Saves 3.6× memory vs f16.

**Prefill expert batching:** Tokens grouped by expert assignment via prefix-sum bucket fill. Each expert dispatched once with all assigned tokens. 28× faster than per-token sequential dispatch.

**No continuous batching:** Single-user target. No request scheduler, no KV cache paging.

## License

Apache-2.0
