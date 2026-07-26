---
type: concept
title: Build & Configuration
date: 2026-07-25
bundle: moe-680m-docs
---

# Build System

## Makefile (shaders)

GLSL → SPIR-V compilation via `glslc` from Vulkan SDK.

```makefile
# Auto-detect all .comp files (skip common.glsl)
COMP_SRCS = $(wildcard shaders/*.comp)
SPV_FILES = $(patsubst shaders/%.comp, src/shaders/%.spv, ...)

# Full rebuild
make all

# Fast iteration (GEMM kernels only)
make gemm
```

All shaders depend on `shaders/common.glsl` — changing it triggers full recompile. SPIR-V files embedded into Rust binary via `include_bytes!()`.

## Cargo.toml

| Dependency | Version | Purpose |
|---|---|---|
| `ash` | 0.38 | Vulkan bindings |
| `memmap2` | 0.9 | GGUF file mmap |
| `bytemuck` | 1.15 | Zeroable/Pod for push constants |
| `libc` | 0.2 | madvise, terminal raw mode |
| `serde` + `serde_json` | 1 | HTTP server JSON (optional) |

### Features

| Feature | Enables |
|---|---|
| `gigatoken` | tiktoken-precise BPE tokenizer (external dep) |
| `server` | Anthropic-compatible HTTP API (`serde` + `serde_json`) |

```bash
# Standard build
cargo build --release

# With server
cargo build --release --features server

# With gigatoken tokenizer
cargo build --release --features gigatoken
```

## Release Profile

```toml
[profile.release]
lto = true
codegen-units = 1
```

LTO enables cross-crate inlining. Single codegen unit maximizes optimization.

## Pipeline Cache

SPIR-V → pipeline compilation cached at `$TMPDIR/moe_pipeline_cache.bin`. Saved after successful `create_pipelines()`. Loaded on next run — skips SPIR-V compilation on subsequent launches. Silent on I/O errors.

## CLI Options

```
moe-680m [OPTIONS]

  --debug              Debug output (or MOE_DEBUG=1)
  --validate           Vulkan validation layers (MOE_VK_VALIDATION=1)
  --smoke              Run Vulkan smoke test (device enumeration + UMA)
  --tui                Interactive TUI configuration (default with no args)
  --chat               Interactive chat mode (multi-turn conversation)
  --model <PATH>       Load GGUF model
  --prompt <TEXT>      Run inference with prompt
  --max-tokens <N>     Max tokens to generate (default 100)
  --server [PORT]      Start HTTP server (default 8080)
  --help               Print help
```

Env vars: `MOE_DEBUG=1`, `MOE_VK_VALIDATION=1`
