# moe-680m — Inference Engine Skill

The skill teaches coding agents (Claude Code, Cursor, Copilot) to work with MoE inference engine for AMD Radeon 680M (RDNA2) APU. Zero to production inference.

---

## Install

### Claude Code / Cursor / Copilot

```
# Point agent to the repo
cd moe-680m
git checkout fused-shaders
```

Include in `CLAUDE.md` or `AGENTS.md`:

```markdown
# moe-680m context
Reference: src/ (Rust + GLSL shaders), shaders/ (20 .comp files),
plans/fused-shaders.md, plans/bugs-found.md
Architecture: MoE 256 experts (8 routed), 40 layers
  (30 DeltaNet + 10 GQA), IQ4_XS quant, Vulkan Compute via ash 0.38
```

### Build

```bash
# Prerequisites: Vulkan SDK (glslc on PATH), Rust nightly
make all           # compile GLSL → SPIR-V
cargo build --release  # build Rust binary
```

### Run

```bash
# Interactive TUI (default)
./target/release/moe-680m

# CLI
./target/release/moe-680m --model model.gguf --prompt "Hello" --max-tokens 200

# Interactive chat
./target/release/moe-680m --model model.gguf --chat
```

---

## What the Engine Does

| Capability | Purpose |
|---|---|
| **Infer** | Run Qwen3.6-35B-A3B MoE model at ~15-20 tok/s on RDNA2 APU |
| **Fuse shaders** | Generation path uses fused kernels (RMS norm + QKV + RoPE, attn + residual) — 46% fewer dispatches |
| **MTP decode** | Speculative decoding with 2 MTP heads, greedy verification pass |
| **Serve** | Anthropic-compatible HTTP API (--features server) |
| **Route** | GPU topk for generation (M=1), CPU prefix-sum batching for prefill (M>1) |
| **Quantize** | IQ4_XS weights (4.5 bpw) + Q4_0 KV cache (inline dequant in shaders) |
| **Chat** | Multi-turn conversation with incremental KV cache |

---

## Architecture

```
┌──────────────────────────────────────────┐
│            Vulkan Compute (ash 0.38)      │
│         Single descriptor, 128B push      │
└────────────────────┬─────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
┌──────────┐   ┌──────────┐   ┌──────────────┐
│ 40 Layers │   │256 Exp   │   │ IQ4_XS + Q4_0│
│ hybrid    │   │8 routed  │   │ 18.2 GB      │
│ DN + GQA  │   │+1 shared │   │ weights      │
└──────────┘   └──────────┘   └──────────────┘
```

### Fused Generation Path (M=1)

```
┌─ GQA Layer (10x) ─────────────────────┐
│ RMSNorm+QKV+RoPE (fused)              │
│   → KV Write → Attention              │
│   → AttnOutput+Residual (fused)       │
│   → Router (GPU topk)                 │
└────────────────────────────────────────┘
```

| Dispatch | Old | Fused | Saved |
|---|---|---|---|
| GQA layer | 10 | 4 | 6 barriers |
| DeltaNet layer | 6 | 4 | 2 dispatches |
| Total per token | 261 | 141 | 120 dispatches |

---

## Usage Examples

### Run inference with TUI

The TUI (`src/tui.rs`) provides arrow-key navigation, inline editing, and parameter setup:

```
┌──────────────────────────────────────┐
│  ↑ Model path: /models/qwen.gguf    │
│  → Prompt: Hello, how are you?       │
│    Temperature: 1.0, Top-K: 0       │
│    Max tokens: 200                   │
│    Interactive chat: ON              │
│  r. Run   s. Smoke   q. Quit        │
└──────────────────────────────────────┘
```

### Add a new fused shader

1. Write `.comp` file in `shaders/` (include `common.glsl`)
2. Add `PipelineType` variant in `src/pipeline.rs`
3. Register with `try_pipeline!` macro
4. Compile: `make all`
5. Wire into `src/inference.rs` `record_and_submit_layer()` with M==1 guard and pipeline-null fallback
6. Push constants reuse existing `PushConstants` struct (128B) — map GLSL fields to Rust byte offsets

### Debug performance

```bash
# Pipeline cache (auto-saved to $TMPDIR)
MOE_DEBUG=1 cargo run --release -- --smoke  # device info, UMA check

# Shader compilation warnings
glslc -fshader-stage=compute -Ishaders shaders/foo.comp -o /dev/null
```

---

## Included Resources

| File | Content |
|---|---|
| `plans/fused-shaders.md` | Fused shader design, dispatch plan, LDS budgets |
| `plans/bugs-found.md` | Bug log: GQA offsets, TUI, chat BOS, DeltaNet state |
| `shaders/common.glsl` | 108 lines shared GLSL: f16→f32 (native RDNA2), IQ4_XS dequant, Q4_0 dequant, SiLU |
| `src/tui.rs` | Zero-dependency interactive TUI (libc raw terminal mode) |

---

## Known Issues

| Issue | File | Status |
|---|---|---|
| GQA K/V offset (×2) | `inference.rs:558` | Fixed — `+4096` → `+8192` |
| DeltaNet gate past buffer | `deltanet_step.comp:48,60` | Pre-existing — gate at element 8192+, buffer has 4128 |
| TUI editing no echo | `tui.rs` | Fixed — restore cooked mode for input |
| Chat BOS missing | `main.rs` | Fixed — prepend BOS in chat loop |

---

## RDNA2-Specific Optimizations

| Technique | Location | Effect |
|---|---|---|
| `unpackHalf2x16` / `packHalf2x16` | `common.glsl` | 1 HW instruction vs 8-10 ALU ops per f16 conversion |
| Wave32 threadgroups (128 threads = 4 waves) | All GEMM shaders | Full wave occupancy, no partial waves |
| `At[64][33]` anti-bank-conflict padding | All GEMM shaders | +1 column avoids 32-bank LDS conflict |
| IQ4_XS dequant inline in GEMM | w1_w3_fused, attn_output, etc. | No separate dequant pass |
| SiLU fused into W2Scatter | `w2_scatter.comp` | Eliminates separate SiLU dispatch |

---

## Links

- Source: `github.com/herlangga72/moe-680m` (branch: `fused-shaders`)
- Driver: Mesa RADV (amdgpu) for RDNA2 UMA support
- AS: `ash 0.38` Vulkan bindings for Rust
- Spec: `plans/fused-shaders.md`
