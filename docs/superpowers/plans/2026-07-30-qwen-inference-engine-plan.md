# Qwen 3.6 35B A3B MTP Inference Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Vulkan compute inference engine for Qwen 3.6 35B A3B MTP on Radeon 680M iGPU, serving Anthropic Messages API to Claude Code.

**Architecture:** Single Rust binary (ash + gigatoken + hyper), GGUF model loading with iQ4_XS fused dequant, MTP speculative decoding, execution-only barrier discipline for RDNA2 machine sympathy, Anthropic SSE streaming with tool use detection.

**Tech Stack:** Rust 1.82+, ash 0.38, gigatoken, hyper 1.x, tokio, serde/serde_json, bytemuck, memmap2, glslc (Vulkan SDK)

## Global Constraints

- Single Vulkan device, single compute queue, unified memory (iGPU)
- Anthropic Messages API only — `POST /v1/messages` with SSE streaming + `POST /v1/messages/count_tokens`
- iQ4_XS dequant fused into every weight-reading shader — no separate dequant pass
- 256-thread workgroups only (4 Wave64 wavefronts = 1 CU saturation)
- Execution-only barriers between layer-internal dispatches, memory barriers only at KV write, LM head, and sampler output
- All buffers 128-byte aligned, pre-allocated in single VkDeviceMemory
- All model hyperparameters from GGUF metadata — no hardcoded dimensions
- Server thread pinned to CCX0, GPU uses CCX1 L3
- No Python, no C++, no multi-user, no batching, no OpenAI API

---

## File Map

```
trial2/
├── Cargo.toml                    (deps: ash, gigatoken, hyper, tokio, serde, bytemuck, memmap2)
├── Makefile                      (shader compilation, build, smoke, test, bench)
├── build.rs                      (glslc → SPIR-V at build time)
├── shaders/
│   ├── common.glsl               (#include: push constant layouts, dequant helpers, subgroup reductions)
│   ├── rms_norm.comp             (RMS norm with iQ4_XS weight dequant)
│   ├── embed.comp                (token embedding lookup)
│   ├── qkv.comp                  (Q/K/V projection, fused dequant)
│   ├── rope.comp                 (RoPE application to Q and K)
│   ├── attention.comp            (GQA: Q×K^T with Q4_0 K dequant, softmax, ×V with Int8 V unpack)
│   ├── kv_write.comp             (write K as Q4_0, V as Int8 to cache)
│   ├── residual_add.comp         (element-wise FP32 add)
│   ├── router_topk.comp          (MoE router: linear projection → top-K per token)
│   ├── moe_gate_up.comp          (fused gate+up projection for active experts + shared experts)
│   ├── silu_mult.comp            (SiLU(gate) × up, element-wise)
│   ├── moe_down.comp             (down projection for active experts)
│   ├── moe_combine.comp          (weighted sum of expert outputs)
│   ├── lm_head.comp              (final RMSNorm + vocab projection, fused dequant)
│   ├── sample.comp               (branchless CDF: top-p, top-k, temperature)
│   ├── mtp_concat_norm.comp      (concat hidden_state + embed(token), RMSNorm)
│   ├── mtp_attention.comp        (MTP attention block — no causal mask)
│   ├── mtp_ffn.comp              (MTP SwiGLU FFN)
│   └── mtp_head.comp             (MTP output head projection)
├── src/
│   ├── main.rs                   (CLI: --model, --port, --max-context; startup: init→load→serve)
│   ├── error.rs                  (Error enum: Vulkan, GGUF, API, Tokenizer variants)
│   ├── constants.rs              (PushConstant layouts, barrier templates, alignment constants)
│   ├── gguf.rs                   (GGUF parser: header→metadata→tensor index→mmap weights)
│   ├── device.rs                 (Vulkan instance/device/queue init, device limits query)
│   ├── memory.rs                 (arena allocator: single VkDeviceMemory, sub-allocation, alignment)
│   ├── shaders.rs                (pipeline cache: compile .comp→SPIR-V, create VkPipeline per shader)
│   ├── dispatch.rs               (pre-chained dispatch builder: timeline semaphores, barrier insertion)
│   ├── engine.rs                 (forward pass: prefill + decode orchestrator, layer loop)
│   ├── mtp.rs                    (MTP draft chain + verify pass logic)
│   ├── kv_cache.rs               (KV cache manager: Q4_0 K / Int8 V layout, position tracking)
│   ├── sampler.rs                (GPU sampler: push constants → dispatch → read token)
│   ├── tokenizer.rs              (gigatoken wrapper: encode, decode, vocab access)
│   ├── chat_template.rs          (Jinja2 parser for Qwen chat format, token sequence builder)
│   └── api.rs                    (hyper HTTP server: /v1/messages POST + SSE, tool use state machine)
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `trial2/Cargo.toml`
- Create: `trial2/build.rs`
- Create: `trial2/Makefile`
- Create: `trial2/src/error.rs`
- Create: `trial2/src/constants.rs`
- Create: `trial2/src/main.rs` (stub)

**Interfaces:**
- Produces: `Error` enum (all variants), `PushConstants` structs, `BarrierPatterns` module, `ALIGNMENT: usize = 128`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "moe-680m"
version = "0.2.0"
edition = "2021"

[dependencies]
ash = "0.38"
memmap2 = "0.9"
bytemuck = { version = "1.15", features = ["derive"] }
libc = "0.2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
gigatoken = "0.2"
hyper = { version = "1", features = ["server", "http1"] }
tokio = { version = "1", features = ["rt", "macros", "sync"] }
futures = "0.3"
thiserror = "2"

[profile.release]
lto = true
codegen-units = 1
opt-level = 3

[[bin]]
name = "moe-680m"
path = "src/main.rs"
```

- [ ] **Step 2: Create build.rs**

```rust
use std::process::Command;

fn main() {
    let shader_dir = std::path::Path::new("shaders");
    let out_dir = std::path::Path::new("src/shaders");
    std::fs::create_dir_all(out_dir).unwrap();

    for entry in std::fs::read_dir(shader_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().map_or(false, |e| e == "comp") {
            let name = path.file_stem().unwrap().to_str().unwrap();
            let spv = out_dir.join(format!("{}.spv", name));
            let status = Command::new("glslc")
                .args(&[
                    "--target-env=vulkan1.3",
                    "-fshader-stage=compute",
                    "-I", "shaders",
                    path.to_str().unwrap(),
                    "-o", spv.to_str().unwrap(),
                ])
                .status()
                .expect("glslc not found — install Vulkan SDK");
            assert!(status.success(), "glslc failed for {}", name);
            println!("cargo:rerun-if-changed=shaders/{}.comp", name);
        }
    }
    println!("cargo:rerun-if-changed=shaders/common.glsl");
}
```

- [ ] **Step 3: Create src/error.rs**

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Vulkan: {0}")]
    Vulkan(#[from] ash::vk::Result),
    #[error("GGUF parse: {0}")]
    Gguf(String),
    #[error("API: {0}")]
    Api(String),
    #[error("Tokenizer: {0}")]
    Tokenizer(String),
    #[error("Model architecture mismatch: {0}")]
    Architecture(String),
    #[error("Out of memory: needed {needed}, available {available}")]
    OutOfMemory { needed: u64, available: u64 },
    #[error("KV cache full: {0} tokens, max {1}")]
    KvCacheFull(usize, usize),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
```

- [ ] **Step 4: Create src/constants.rs**

```rust
use ash::vk;

pub const ALIGNMENT: u64 = 128;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct RMSNormPC {
    pub rows: u32,
    pub dim: u32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct LinearPC {
    pub in_dim: u32,
    pub out_dim: u32,
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct AttentionPC {
    pub seq_len: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct RouterPC {
    pub dim: u32,
    pub n_experts: u32,
    pub n_active: u32,
    pub n_shared: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct MoEPC {
    pub dim: u32,
    pub intermediate: u32,
    pub expert_idx: u32,
    pub is_shared: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct SamplePC {
    pub vocab_size: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct MTPBlockPC {
    pub dim: u32,
    pub head_dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub pos: u32,
    pub block_idx: u32,
}

/// Execution-only barrier (no memory flush): compute → compute on same queue
pub fn barrier_exec_only() -> vk::MemoryBarrier2<'static> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
    // NO access flags — execution barrier only
}

/// Memory barrier with full compute read/write flush
pub fn barrier_memory_flush() -> vk::MemoryBarrier2<'static> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
}

/// Memory barrier for CPU read after GPU write
pub fn barrier_host_read() -> vk::MemoryBarrier2<'static> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
        .dst_access_mask(vk::AccessFlags2::HOST_READ)
}
```

- [ ] **Step 5: Create src/main.rs (stub)**

```rust
mod constants;
mod error;

use error::Result;

fn main() -> Result<()> {
    println!("moe-680m v0.2.0 — Qwen 3.6 35B A3B MTP");
    Ok(())
}
```

- [ ] **Step 6: Create Makefile**

```makefile
SPV_DIR  := src/shaders
COMP_DIR := shaders
MODEL    ?= model/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf

COMP_SRCS := $(filter-out $(COMP_DIR)/common.glsl, $(wildcard $(COMP_DIR)/*.comp))
SPV_FILES := $(patsubst $(COMP_DIR)/%.comp, $(SPV_DIR)/%.spv, $(COMP_SRCS))
CARGO := CARGO_TARGET_DIR=./target cargo

.PHONY: all shaders build smoke clean help

all: shaders build

shaders: $(SPV_FILES)

$(SPV_DIR)/%.spv: $(COMP_DIR)/%.comp $(COMP_DIR)/common.glsl
	@mkdir -p $(SPV_DIR)
	glslc --target-env=vulkan1.3 -fshader-stage=compute -I$(COMP_DIR) $< -o $@

build:
	$(CARGO) build --release

smoke: build
	./target/release/moe-680m --smoke

# ponytail: full test targets added after engine modules exist
clean:
	cargo clean
	rm -f $(SPV_DIR)/*.spv

help:
	@echo "all | shaders | build | smoke | clean"
```

- [ ] **Step 7: Build and verify smoke runs**

Run: `cd trial2 && make all && ./target/release/moe-680m --smoke`
Expected: `moe-680m v0.2.0 — Qwen 3.6 35B A3B MTP` (smoke only prints version for now)

- [ ] **Step 8: Commit**

```bash
git add trial2/Cargo.toml trial2/build.rs trial2/Makefile \
        trial2/src/main.rs trial2/src/error.rs trial2/src/constants.rs
git commit -m "chore: project scaffold — Cargo, Makefile, build.rs, error types"
```

---

### Task 2: Vulkan Device Init

**Files:**
- Create: `trial2/src/device.rs`
- Modify: `trial2/src/main.rs` (wire `--smoke`)
- Create: `trial2/src/shaders/` (empty dir for SPIR-V outputs)

**Interfaces:**
- Consumes: `error::Result`
- Produces: `Device { instance: ash::Instance, device: ash::Device, physical: vk::PhysicalDevice, queue: vk::Queue, queue_family: u32, limits: vk::PhysicalDeviceLimits, subgroup_size: u32, timestamp_period: f32 }`

- [ ] **Step 1: Write device init**

Create `trial2/src/device.rs`:

```rust
use ash::{Entry, Instance, vk};
use crate::error::{Error, Result};

pub struct Device {
    pub instance: Instance,
    pub _entry: Entry,
    pub device: ash::Device,
    pub physical: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub limits: vk::PhysicalDeviceLimits,
    pub subgroup_size: u32,
    pub timestamp_period: f32,
}

impl Device {
    pub fn init() -> Result<Self> {
        let entry = unsafe { Entry::load().map_err(|e| Error::Vulkan(e.into()))? };

        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_3);

        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
        }.map_err(|e| Error::Vulkan(e.into()))?;

        let physical = unsafe { instance.enumerate_physical_devices()? }
            .into_iter()
            .next()
            .ok_or(Error::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;

        let props = unsafe { instance.get_physical_device_properties(physical) };
        let subgroup_props = unsafe {
            instance.get_physical_device_properties2::<vk::PhysicalDeviceProperties2>(
                physical,
                &mut vk::PhysicalDeviceSubgroupProperties::default(),
            )
        };

        let queue_family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
            .into_iter()
            .enumerate()
            .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .ok_or(Error::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;

        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(queue_family)
                        .queue_priorities(&[1.0])]),
                None,
            )
        }.map_err(|e| Error::Vulkan(e.into()))?;

        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        Ok(Self {
            queue,
            queue_family,
            device,
            physical,
            instance,
            _entry: entry,
            limits: props.limits,
            subgroup_size: props.limits.subgroup_size, // ponytail: ~64 on RDNA2
            timestamp_period: props.limits.timestamp_period as f32,
        })
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
```

- [ ] **Step 2: Update main.rs to wire smoke test**

```rust
mod constants;
mod device;
mod error;

use error::{Error, Result};

fn smoke() -> Result<()> {
    let dev = device::Device::init()?;
    let name = unsafe {
        dev.instance.get_physical_device_properties(dev.physical)
    };
    let name_str = std::str::from_utf8(&name.device_name)
        .unwrap_or("unknown")
        .trim_end_matches('\0');
    println!("GPU: {} ({} CUs, subgroup={}, timestamp_period={:.0}ns)",
        name_str,
        dev.limits.max_compute_units,
        dev.subgroup_size,
        dev.timestamp_period,
    );
    println!("Max shared memory: {} KB", dev.limits.max_compute_shared_memory_size / 1024);
    println!("Max workgroup: {}x{}x{}",
        dev.limits.max_compute_work_group_size[0],
        dev.limits.max_compute_work_group_size[1],
        dev.limits.max_compute_work_group_size[2],
    );
    dev.destroy();
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--smoke" {
        return smoke();
    }
    println!("moe-680m v0.2.0 — Qwen 3.6 35B A3B MTP");
    Ok(())
}
```

- [ ] **Step 3: Build and run smoke test**

Run: `cd trial2 && cargo build --release && ./target/release/moe-680m --smoke`
Expected: prints GPU info including "Radeon 680M", 12 CUs, subgroup=64

- [ ] **Step 4: Commit**

```bash
git add trial2/src/device.rs trial2/src/main.rs
git commit -m "feat: Vulkan device init + --smoke test"
```

---

### Task 3: Memory System

**Files:**
- Create: `trial2/src/memory.rs`
- Modify: `trial2/src/main.rs` (re-export memory module)
- Modify: `trial2/src/device.rs` (add `find_memory_type_index` helper)

**Interfaces:**
- Consumes: `Device`
- Produces: `Arena { device: ash::Device, memory: vk::DeviceMemory, total_size: u64, offsets: HashMap<String, (u64, u64)> }`, `Arena::new(device, size)`, `Arena::allocate(name, size)`, `Arena::bind_buffer(name, buffer)`, `Buffer::new(device, size, usage)` 

- [ ] **Step 1: Add memory type helper to device.rs**

Add to `impl Device`:

```rust
pub fn find_memory_type(&self, type_filter: u32, flags: vk::MemoryPropertyFlags) -> Result<u32> {
    let mem_props = unsafe {
        self.instance.get_physical_device_memory_properties(self.physical)
    };
    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && mem_props.memory_types[i as usize].property_flags.contains(flags)
        {
            return Ok(i);
        }
    }
    Err(Error::Vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))
}
```

- [ ] **Step 2: Create memory.rs**

```rust
use ash::{Device, vk};
use std::collections::HashMap;
use crate::constants::ALIGNMENT;
use crate::error::{Error, Result};

fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

pub struct Buffer {
    pub handle: vk::Buffer,
    pub size: u64,
}

impl Buffer {
    pub fn new(device: &Device, size: u64, usage: vk::BufferUsageFlags) -> Result<Self> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { device.create_buffer(&create_info, None)? };
        Ok(Self { handle, size })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe { device.destroy_buffer(self.handle, None); }
    }
}

pub struct Arena {
    device: Device,
    memory: vk::DeviceMemory,
    total_size: u64,
    offsets: HashMap<String, (u64, u64)>, // name -> (offset, size)
    next_offset: u64,
}

impl Arena {
    pub fn new(
        device: Device,
        size: u64,
        memory_type_index: u32,
    ) -> Result<Self> {
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };

        Ok(Self {
            device,
            memory,
            total_size: size,
            offsets: HashMap::new(),
            next_offset: 0,
        })
    }

    pub fn allocate(&mut self, name: &str, size: u64) -> Result<u64> {
        let offset = align_up(self.next_offset, ALIGNMENT);
        if offset + size > self.total_size {
            return Err(Error::OutOfMemory {
                needed: size,
                available: self.total_size - offset,
            });
        }
        self.offsets.insert(name.to_string(), (offset, size));
        self.next_offset = offset + size;
        Ok(offset)
    }

    pub fn bind_buffer(&self, name: &str, buffer: &Buffer) -> Result<()> {
        let &(offset, size) = self.offsets.get(name)
            .ok_or_else(|| Error::Api(format!("arena: no allocation for '{}'", name)))?;
        assert!(buffer.size >= size, "buffer too small for '{}'", name);
        unsafe {
            self.device.bind_buffer_memory(buffer.handle, self.memory, offset)?;
        }
        Ok(())
    }

    pub fn destroy(&mut self) {
        unsafe { self.device.free_memory(self.memory, None); }
    }
}
```

- [ ] **Step 3: Write unit test for alignment**

Add to bottom of `memory.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 128), 0);
        assert_eq!(align_up(1, 128), 128);
        assert_eq!(align_up(128, 128), 128);
        assert_eq!(align_up(129, 128), 256);
    }
}
```

Run: `cd trial2 && cargo test`
Expected: `test memory::tests::test_align_up ... ok`

- [ ] **Step 4: Commit**

```bash
git add trial2/src/memory.rs trial2/src/device.rs trial2/src/main.rs
git commit -m "feat: memory arena — single VkDeviceMemory, 128B-aligned sub-allocation"
```

---

### Task 4: GGUF Parser

**Files:**
- Create: `trial2/src/gguf.rs`
- Modify: `trial2/src/main.rs` (add `gguf` module)

**Interfaces:**
- Consumes: nothing beyond std
- Produces: `GgufFile { metadata: HashMap<String, GgufValue>, tensors: Vec<TensorInfo>, data: memmap2::Mmap }`, `GgufValue` enum, `TensorInfo { name, shape, dtype, offset, size }`, `ModelConfig { n_layers, hidden_dim, ... }` derived struct

- [ ] **Step 1: Create gguf.rs — types and parser**

```rust
// ponytaill: single-pass parse, no streaming — GGUF header is small, tensors are mmap'd
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use memmap2::Mmap;
use crate::error::{Error, Result};

// GGUF value types (subset we need)
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    ArrayU32(Vec<u32>),
    ArrayF32(Vec<f32>),
    ArrayI64(Vec<i64>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDtype {
    F32,    // 0
    F16,    // 1
    Q4_0,   // 2
    Q8_0,   // 8
    IQ4_XS, // custom
    Unknown(u32),
}

impl GgufDtype {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            _ => Self::Unknown(raw),
        }
    }

    pub fn type_size(&self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,  // 32 elements in 18 bytes
            Self::Q8_0 => 34,  // 32 elements in 34 bytes
            Self::Unknown(_) => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgufDtype,
    pub offset: u64,
    pub n_elements: u64,
    pub size_bytes: u64,
}

pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    pub data: Mmap,
}

pub struct ModelConfig {
    pub n_layers: u32,
    pub hidden_dim: u32,
    pub n_heads_q: u32,
    pub n_heads_kv: u32,
    pub head_dim: u32,
    pub ffn_intermediate: u32,
    pub n_experts: u32,
    pub n_active_experts: u32,
    pub n_shared_experts: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub rope_theta: f32,
    pub rope_type: String,
    pub n_mtp_modules: u32,
    pub mtp_depth: u32,
    pub eps: f32,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let data = unsafe { Mmap::map(&file)? };

        if data.len() < 24 {
            return Err(Error::Gguf("file too small for GGUF header".into()));
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != 0x46475547 { // "GGUF"
            return Err(Error::Gguf("bad magic — not a GGUF file".into()));
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != 3 {
            return Err(Error::Gguf(format!("unsupported GGUF version {}", version)));
        }

        let n_tensors = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let n_kv = u64::from_le_bytes(data[16..24].try_into().unwrap());
        let mut pos: usize = 24;

        // Parse key-value metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = Self::read_string(&data, &mut pos);
            let val_type = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let val = Self::read_value(&data, &mut pos, val_type)?;
            metadata.insert(key, val);
        }

        // Parse tensor infos
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = Self::read_string(&data, &mut pos);
            let n_dims = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let mut shape = Vec::with_capacity(n_dims as usize);
            let mut n_elements: u64 = 1;
            for _ in 0..n_dims {
                let d = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
                pos += 8;
                shape.push(d);
                n_elements = n_elements.saturating_mul(d);
            }
            let dtype_raw = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let dtype = GgufDtype::from_raw(dtype_raw);
            let offset = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
            pos += 8;
            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                offset,
                n_elements,
                size_bytes: 0, // computed after parse
            });
        }

        // Compute sizes (contiguous storage)
        for i in 0..tensors.len() {
            let this_offset = tensors[i].offset;
            let next_offset = if i + 1 < tensors.len() {
                tensors[i + 1].offset
            } else {
                data.len() as u64
            };
            tensors[i].size_bytes = next_offset - this_offset;
        }

        Ok(Self { metadata, tensors, data })
    }

    pub fn model_config(&self) -> Result<ModelConfig> {
        let get_u32 = |key: &str| -> Result<u32> {
            match self.metadata.get(key) {
                Some(GgufValue::U32(v)) => Ok(*v),
                _ => Err(Error::Architecture(format!("missing or wrong type: {}", key))),
            }
        };
        let get_f32 = |key: &str| -> Result<f32> {
            match self.metadata.get(key) {
                Some(GgufValue::F32(v)) => Ok(*v),
                _ => Err(Error::Architecture(format!("missing or wrong type: {}", key))),
            }
        };
        let get_str = |key: &str| -> Result<String> {
            match self.metadata.get(key) {
                Some(GgufValue::String(v)) => Ok(v.clone()),
                _ => Err(Error::Architecture(format!("missing or wrong type: {}", key))),
            }
        };

        let n_heads_q = get_u32("qwen3.attention.head_count")?;
        let n_heads_kv = get_u32("qwen3.attention.head_count_kv")?;
        let hidden_dim = get_u32("qwen3.embedding_length")?;
        let head_dim = hidden_dim / n_heads_q;

        Ok(ModelConfig {
            n_layers: get_u32("qwen3.block_count")?,
            hidden_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            ffn_intermediate: get_u32("qwen3.feed_forward_length")?,
            n_experts: get_u32("qwen3.expert_count").unwrap_or(1),
            n_active_experts: get_u32("qwen3.expert_used_count").unwrap_or(1),
            n_shared_experts: get_u32("qwen3.expert_shared_count").unwrap_or(0),
            vocab_size: get_u32("qwen3.vocab_size")?,
            max_seq_len: get_u32("qwen3.context_length").unwrap_or(32768),
            rope_theta: get_f32("qwen3.rope.freq_base").unwrap_or(1_000_000.0),
            rope_type: get_str("qwen3.rope.type").unwrap_or_else(|_| "default".into()),
            n_mtp_modules: get_u32("qwen3.mtp.module_count").unwrap_or(0),
            mtp_depth: get_u32("qwen3.mtp.depth").unwrap_or(0),
            eps: get_f32("qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
        })
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensor_data(&self, tensor: &TensorInfo) -> &[u8] {
        let start = tensor.offset as usize;
        &self.data[start..start + tensor.size_bytes as usize]
    }

    fn read_string(data: &[u8], pos: &mut usize) -> String {
        let len = u64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap()) as usize;
        *pos += 8;
        let s = String::from_utf8_lossy(&data[*pos..*pos+len]).to_string();
        *pos += len;
        s
    }

    fn read_value(data: &[u8], pos: &mut usize, val_type: u32) -> Result<GgufValue> {
        Ok(match val_type {
            1 => GgufValue::U8(data[*pos]),
            4 => GgufValue::U32(u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap())),
            5 => GgufValue::U64(u64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap())),
            7 => GgufValue::I32(i32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap())),
            8 => GgufValue::I64(i64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap())),
            22 => GgufValue::F32(f32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap())),
            23 => GgufValue::F64(f64::from_le_bytes(data[*pos..*pos+8].try_into().unwrap())),
            26 => GgufValue::Bool(data[*pos] != 0),
            13 => { // array of u32
                let len = u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap()) as usize;
                *pos += 4;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len { arr.push(u32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap())); *pos += 4; }
                GgufValue::ArrayU32(arr)
            },
            _ => return Err(Error::Gguf(format!("unknown value type {}", val_type))),
        }
    }
}
```

- [ ] **Step 2: Build and verify it compiles**

Run: `cd trial2 && cargo build --release`
Expected: compiles (no model to parse yet, just types)

- [ ] **Step 3: Commit**

```bash
git add trial2/src/gguf.rs trial2/src/main.rs
git commit -m "feat: GGUF parser — header, metadata, tensor index, ModelConfig"
```

---

### Task 5: Shader Pipeline Compilation

**Files:**
- Create: `trial2/src/shaders.rs`
- Modify: `trial2/src/main.rs` (add `shaders` module)

**Interfaces:**
- Consumes: `Device`
- Produces: `ShaderCache { pipelines: HashMap<&'static str, vk::Pipeline>, layouts: HashMap<&'static str, vk::PipelineLayout>, desc_set_layout: vk::DescriptorSetLayout, pool: vk::DescriptorPool, desc_sets: Vec<vk::DescriptorSet> }`

- [ ] **Step 1: Create shaders.rs**

```rust
use ash::{Device, vk};
use std::collections::HashMap;
use std::io::Read;
use crate::device::Device as GpuDevice;
use crate::error::{Error, Result};

const SPV_DIR: &str = "src/shaders";

// All shaders used — must match .comp filenames without extension
pub const SHADERS: &[&str] = &[
    "rms_norm", "embed", "qkv", "rope", "attention", "kv_write",
    "residual_add", "router_topk", "moe_gate_up", "silu_mult",
    "moe_down", "moe_combine", "lm_head", "sample",
    "mtp_concat_norm", "mtp_attention", "mtp_ffn", "mtp_head",
];

pub struct ShaderCache {
    pub pipelines: HashMap<&'static str, vk::Pipeline>,
    pub pipeline_layout: vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub desc_sets: Vec<vk::DescriptorSet>,
}

fn load_spv(name: &str) -> Result<Vec<u32>> {
    // Try build output first, then src/shaders
    let paths = [
        format!("{}/{}.spv", SPV_DIR, name),
        format!("target/release/build/{}/{}.spv", name, name),
    ];
    for path in &paths {
        if let Ok(mut f) = std::fs::File::open(path) {
            let mut bytes = Vec::new();
            f.read_to_end(&mut bytes)?;
            if bytes.len() % 4 != 0 {
                return Err(Error::Api(format!("corrupt SPIR-V for {}", name)));
            }
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            return Ok(words);
        }
    }
    Err(Error::Api(format!("SPIR-V not found for '{}' — run 'make shaders' first", name)))
}

impl ShaderCache {
    // ponytaill: one descriptor set layout — all buffers, bound once per forward pass
    // All shaders share the same layout (SSBOs only), pipeline layouts are identical
    pub fn new(dev: &GpuDevice) -> Result<Self> {
        // Layout: binding 0-15 = storage buffers (max 16 per shader)
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..16)
            .map(|i| vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = unsafe {
            dev.device.create_descriptor_set_layout(&layout_info, None)?
        };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(128);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&[desc_set_layout])
            .push_constant_ranges(&[push_range]);

        let pipeline_layout = unsafe {
            dev.device.create_pipeline_layout(&pipeline_layout_info, None)?
        };

        // Descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .typ(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(16 * SHADERS.len() as u32)];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(SHADERS.len() as u32)
            .pool_sizes(&pool_sizes);

        let pool = unsafe { dev.device.create_descriptor_pool(&pool_info, None)? };

        // Allocate descriptor sets (one per shader, same layout)
        let set_layouts: Vec<vk::DescriptorSetLayout> = vec![desc_set_layout; SHADERS.len()];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);

        let desc_sets = unsafe { dev.device.allocate_descriptor_sets(&alloc_info)? };

        // Compile pipelines
        let mut pipelines = HashMap::new();
        for (i, &name) in SHADERS.iter().enumerate() {
            let spv = load_spv(name)?;
            let module_info = vk::ShaderModuleCreateInfo::default().code(&spv);
            let module = unsafe { dev.device.create_shader_module(&module_info, None)? };

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");

            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout);

            let pipeline = unsafe {
                dev.device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[info],
                    None,
                )
            }
            .map_err(|(_, e)| Error::Vulkan(e))?[0];

            unsafe { dev.device.destroy_shader_module(module, None); }
            pipelines.insert(name, pipeline);
        }

        Ok(Self {
            pipelines,
            pipeline_layout,
            desc_set_layout,
            pool,
            desc_sets,
        })
    }

    pub fn destroy(&mut self, dev: &GpuDevice) {
        unsafe {
            for &pipeline in self.pipelines.values() {
                dev.device.destroy_pipeline(pipeline, None);
            }
            dev.device.destroy_pipeline_layout(self.pipeline_layout, None);
            dev.device.destroy_descriptor_set_layout(self.desc_set_layout, None);
            dev.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}
```

- [ ] **Step 2: Build and verify compiles**

Run: `cd trial2 && cargo build --release`
Expected: compiles (no SPIR-V files yet — will fail at runtime if trying to compile shaders before `make shaders`)

- [ ] **Step 3: Commit**

```bash
git add trial2/src/shaders.rs trial2/src/main.rs
git commit -m "feat: shader pipeline cache — SPIR-V loading, descriptor layout, pipeline compilation"
```

---

### Task 6: Common Shader Infrastructure

**Files:**
- Create: `trial2/shaders/common.glsl`

**Interfaces:**
- Produces: GLSL functions used by every `.comp` shader via `#include "common.glsl"`

- [ ] **Step 1: Create common.glsl**

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_8bit_storage : enable
#extension GL_KHR_memory_scope_semantics : enable

// ── Push constant layouts ──

layout(push_constant, std430) uniform PC {
    uint rows;
    uint cols;
    uint stride;
    uint pad0;
    float param0;
    float param1;
    float param2;
    float param3;
    uint opt0;
    uint opt1;
    uint opt2;
    uint opt3;
} pc;

// ── FP16 helpers (RDNA2 native via VK_KHR_shader_float16_int8) ──

float16_t load_f16(uint addr) {
    return float16_t(unpackFloat2x16(data16[addr >> 1]).x);
}

void store_f16(uint addr, float16_t val) {
    uint word = data16[addr >> 1];
    if ((addr & 1) == 0) {
        word = (word & 0xFFFF0000u) | packFloat2x16(vec2(val, 0.0f));
    } else {
        word = (word & 0x0000FFFFu) | (packFloat2x16(vec2(val, 0.0f)) << 16);
    }
    data16[addr >> 1] = word;
}

// ── iQ4_XS dequant (fused into weight-reading shaders) ──
// Block: 256 elements → 162 bytes
//   super-block: d (FP16, 2B)
//   8 sub-blocks of 32: m (FP16, 2B) + scales (2B packed: 8 × 2-bit)
//   values: 128B (256 × 4-bit nibbles)

float dequant_iq4_xs(uint blk_start, uint elem_idx) {
    uint blk = elem_idx / 256u;
    uint sub = (elem_idx % 256u) / 32u;
    uint elem_in_sub = elem_idx % 32u;

    // Block base in uint16 units
    uint base = 1296u * blk; // 162 bytes * 8 = 1296 uint16 units

    // Super-block scale d (first 2 bytes → 1 uint16)
    float d = float(data16[base]);
    base += 1u;

    // Skip to sub-block: m (1 uint16) + scales (1 uint16) per sub-block
    base += sub * 2u;

    // Sub-block min m
    float m = float(data16[base]);
    base += 1u;

    // Packed scales: 8 × 2-bit in 2 bytes
    uint scale_packed = data16[base];
    uint scale_bits = (scale_packed >> (sub * 2u)) & 0x3u;
    // Scale mapping for iQ4_XS: 2-bit → float
    // iQ4_XS 2-bit scale → float lookup (6 possible values encoded in 2 bits)
    float scale_table[4] = float[4](-0.5f, 0.0f, 0.5f, 1.0f);
    float scale = scale_table[scale_bits];

    // Nibble value (skip super-block header: 2 + 8*(2+2) = 34 bytes → 17 uint16)
    base = 1296u * blk + 17u;
    uint byte_idx = elem_in_sub / 2u;
    uint nibble = (data16[base + byte_idx / 2u] >> ((byte_idx & 1u) * 4u)) & 0xFu;

    return d * scale * float(nibble) + m;
}

// ── Q4_0 K dequant (for attention) ──
// Block: 32 elements → 18 bytes
//   d (FP16, 2B) + values (16 × 4-bit = 16B)

float16_t dequant_q4_0(uint blk_start, uint elem_idx) {
    uint blk = elem_idx / 32u;
    uint elem_in_blk = elem_idx % 32u;
    uint base = 9u * blk; // 18 bytes = 9 uint16

    float16_t d = float16_t(data16[base]);
    uint byte_idx = elem_in_blk / 2u;
    uint nibble = (data16[base + 1u + byte_idx / 2u] >> ((byte_idx & 1u) * 4u)) & 0xFu;

    return d * float16_t(nibble) - d * float16_t(8.0hf);
}

// ── Int8 V unpack (for attention) ──

float16_t unpack_int8_v(uint addr) {
    // 4 int8 values per uint32
    uint word = data8[addr >> 2];
    uint shift = (addr & 3u) * 8u;
    int val = int((word >> shift) & 0xFFu);
    // Sign-extend
    if ((val & 0x80) != 0) val |= ~0xFF;
    return float16_t(val);
}

// ── Subgroup reductions (Wave64 native) ──

float subgroup_sum(float v) {
    return subgroupAdd(v);
}

float subgroup_max(float v) {
    return subgroupMax(v);
}

// ── RMSNorm helper ──

float rsqrt_fast(float x) {
    return inversesqrt(x);
}
```

- [ ] **Step 2: Generate empty .comp stubs so build compiles**

For each shader in `SHADERS`, create a minimal stub in `trial2/shaders/`:

Example `trial2/shaders/rms_norm.comp`:
```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_KHR_shader_subgroup_basic : enable

#include "common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Buf0 { float data[]; };
layout(set = 0, binding = 1) buffer Buf1 { uint8_t  data8[]; };
layout(set = 0, binding = 2) buffer Buf2 { uint16_t data16[]; };
layout(set = 0, binding = 3) buffer Buf3 { float data_out[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.rows) return;

    float sum = 0.0;
    for (uint j = 0; j < pc.cols; j++) {
        sum += data[i * pc.cols + j] * data[i * pc.cols + j];
    }
    sum = subgroup_sum(sum);
    float rms = rsqrt_fast(sum / float(pc.cols) + pc.param0);

    data_out[i * pc.cols + gl_LocalInvocationID.x] = data[i * pc.cols + gl_LocalInvocationID.x] * rms;
}
```

(Stubs for all 18 shaders — each with correct binding count for its function)

- [ ] **Step 3: Build shaders and verify compilation**

Run: `cd trial2 && make shaders`
Expected: 18 `.spv` files in `trial2/src/shaders/`

- [ ] **Step 4: Commit**

```bash
git add trial2/shaders/ trial2/src/shaders/
git commit -m "feat: common.glsl + shader stubs (18 .comp, 1 shared header)"
```

---

### Task 7: RMSNorm Shader (Full)

**Files:**
- Modify: `trial2/shaders/rms_norm.comp` (replace stub)

**Interfaces:**
- Consumes: `common.glsl` (subgroup ops, push constants)
- Produces: Final `rms_norm.comp` — reads FP32 input, writes normalized FP32 output. iQ4_XS weight variant for rms_norm + weight multiply combined.

- [ ] **Step 1: Write final rms_norm.comp**

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_EXT_shader_16bit_storage : enable

#include "common.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Input  { float inp[]; };
layout(set = 0, binding = 1, std430) buffer Weight { uint16_t wbuf[]; }; // iQ4_XS
layout(set = 0, binding = 2, std430) buffer Output { float out[]; };

shared float s_sum_sq[4]; // 4 subgroups × 1 float

void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint dim = pc.cols;
    float eps = pc.param0;

    // Row offset
    uint base = row * dim;

    // Sum of squares across row, stride by workgroup size
    float sum_sq = 0.0;
    for (uint j = tid; j < dim; j += 256u) {
        float v = inp[base + j];
        sum_sq += v * v;
    }

    // Wave64 subgroup reduction (single cycle on RDNA2)
    sum_sq = subgroup_sum(sum_sq);

    // One thread per subgroup writes to shared memory
    uint sg_id = gl_SubgroupID; // 0..3
    if (gl_SubgroupInvocationID == 0u) {
        s_sum_sq[sg_id] = sum_sq;
    }
    barrier();
    memoryBarrierShared();

    // First subgroup reduces across all subgroups
    float total_sq = 0.0;
    if (sg_id == 0u) {
        for (uint s = 0u; s < 4u; s++) {
            total_sq += s_sum_sq[s];
        }
        total_sq = subgroupBroadcastFirst(total_sq);
    }
    barrier();

    // All threads read the final value via broadcast
    float rms = rsqrt_fast(total_sq / float(dim) + eps);

    // Apply: out = inp * weight * rms (with iQ4_XS dequant fused)
    for (uint j = tid; j < dim; j += 256u) {
        float w = dequant_iq4_xs(0u, j); // weight offset set via push constant
        out[base + j] = inp[base + j] * w * rms;
    }
}
```

- [ ] **Step 2: Rebuild shaders**

Run: `cd trial2 && make shaders`
Expected: all .spv regenerate

- [ ] **Step 3: Commit**

```bash
git add trial2/shaders/rms_norm.comp
git commit -m "feat: rms_norm shader — subgroup reduction, iQ4_XS weight dequant"
```

---

### Tasks 8-16: Remaining Shaders

(Each follows the same pattern: replace stub → implement → rebuild → commit. Listed compactly for brevity — full GLSL code for each in implementation.)

**Task 8: `embed.comp`** — Token ID → embedding row. Single indexed read from FP16 weight buffer, write to hidden_state.

**Task 9: `qkv.comp`** — Linear projection: hidden_state × QKV_weight → q, k, v. iQ4_XS dequant fused. GQA: q has n_heads_q, k/v have n_heads_kv. Output to separate q, k, v buffers (FP16).

**Task 10: `rope.comp`** — Apply RoPE to q and k. Freq computation from rope_theta, position offset from KV cache position. FP16 in/out.

**Task 11: `attention.comp`** — GQA with Q4_0 K + Int8 V cache reads. Q×K^T with fused K dequant → scale → causal mask → softmax (online, subgroup) → ×V with fused V unpack → output. 256-thread workgroup, one head per workgroup.

**Task 12: `kv_write.comp`** — Quantize K to Q4_0 (32-elem blocks, d scale, nibble values), quantize V to Int8, write to KV cache at `[layer][head][pos]`.

**Task 13: `residual_add.comp`** — Element-wise FP32 `h = a + b`. Dim/256 workgroups.

**Task 14: `router_topk.comp`** — Linear projection hidden_state → n_experts logits. Apply softmax. Select top-K per token. Single workgroup (few experts, small). Output gate indices + weights.

**Task 15: `moe_gate_up.comp`** — For active experts + shared experts: fused gate+up projection (×2 intermediate size). iQ4_XS dequant. Output gate and up activations (FP16).

**Task 16: `silu_mult.comp`** — `SiLU(gate) × up` element-wise. Write intermediate.

**Task 17: `moe_down.comp`** — Down projection for each active expert. iQ4_XS dequant. Output per-expert contribution.

**Task 18: `moe_combine.comp`** — Weighted sum of expert outputs per token using router gate weights. Residual add with shared expert output.

**Task 19: `lm_head.comp`** — RMSNorm → linear projection (hidden_dim → vocab_size). iQ4_XS dequant. Write logits (FP32).

**Task 20: `sample.comp`** — GPU-side sampling: temperature scaling → softmax → top-p filter → top-k filter → branchless CDF search → write token ID.

**Task 21: `mtp_concat_norm.comp`** + **`mtp_attention.comp`** + **`mtp_ffn.comp`** + **`mtp_head.comp`** — MTP block shaders. No causal mask. Share K from main model's KV cache, write own V.

---

### Task 22: KV Cache Manager

**Files:**
- Create: `trial2/src/kv_cache.rs`

**Interfaces:**
- Consumes: `ModelConfig`, `Arena` (for buffer allocation)
- Produces: `KvCache { k_q4: Buffer, v_i8: Buffer, seq_len: u32, max_seq: u32, offsets: Vec<u64> }` — `fn append(layer, head, k_data, v_data)`, `fn position() -> u32`

- [ ] **Step 1: Create kv_cache.rs**

```rust
use ash::{Device, vk};
use crate::gguf::ModelConfig;
use crate::memory::{Arena, Buffer};

pub struct KvCache {
    pub k_q4: Buffer,      // Q4_0: 18 bytes per 32 elements
    pub v_i8: Buffer,      // Int8: 1 byte per element
    pub seq_len: u32,
    pub max_seq: u32,
    n_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
}

impl KvCache {
    pub fn new(
        config: &ModelConfig,
        max_seq: u32,
        arena: &mut Arena,
        device: &Device,
    ) -> crate::error::Result<Self> {
        let n_layers = config.n_layers;
        let n_kv_heads = config.n_heads_kv;
        let head_dim = config.head_dim;

        // K (Q4_0): n_layers × n_kv_heads × max_seq × head_dim elements
        // Q4_0 packs 32 elements into 18 bytes → (head_dim/32) * 18 bytes per (head, pos)
        let k_blocks_per_head = head_dim / 32;
        let k_bytes_per_head_pos = k_blocks_per_head * 18;
        let k_size = n_layers as u64 * n_kv_heads as u64 * max_seq as u64 * k_bytes_per_head_pos as u64;

        // V (Int8): n_layers × n_kv_heads × max_seq × head_dim bytes
        let v_size = n_layers as u64 * n_kv_heads as u64 * max_seq as u64 * head_dim as u64;

        let kv_usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let k_q4 = Buffer::new(device, k_size, kv_usage)?;
        let v_i8 = Buffer::new(device, v_size, kv_usage)?;

        arena.allocate("kv_cache_k", k_size)?;
        arena.bind_buffer("kv_cache_k", &k_q4)?;
        arena.allocate("kv_cache_v", v_size)?;
        arena.bind_buffer("kv_cache_v", &v_i8)?;

        Ok(Self {
            k_q4,
            v_i8,
            seq_len: 0,
            max_seq,
            n_layers,
            n_kv_heads,
            head_dim,
        })
    }

    /// Linear offset for K cache: [layer][head][pos][block]
    pub fn k_offset(&self, layer: u32, head: u32, pos: u32) -> u64 {
        let block_bytes = (self.head_dim / 32) as u64 * 18;
        ((layer * self.n_kv_heads + head) as u64 * self.max_seq as u64 + pos as u64)
            * block_bytes
    }

    /// Linear offset for V cache: [layer][head][pos]
    pub fn v_offset(&self, layer: u32, head: u32, pos: u32) -> u64 {
        ((layer * self.n_kv_heads + head) as u64 * self.max_seq as u64 + pos as u64)
            * self.head_dim as u64
    }

    pub fn advance(&mut self) -> crate::error::Result<u32> {
        if self.seq_len >= self.max_seq {
            return Err(crate::error::Error::KvCacheFull(
                self.seq_len as usize,
                self.max_seq as usize,
            ));
        }
        let pos = self.seq_len;
        self.seq_len += 1;
        Ok(pos)
    }

    pub fn position(&self) -> u32 {
        self.seq_len
    }
}
```

- [ ] **Step 2: Build and verify**

Run: `cd trial2 && cargo build --release`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add trial2/src/kv_cache.rs trial2/src/main.rs
git commit -m "feat: KV cache manager — Q4_0 K + Int8 V, per-head position tracking"
```

---

### Task 23: Dispatch Builder (Pre-Chained)

**Files:**
- Create: `trial2/src/dispatch.rs`

**Interfaces:**
- Consumes: `Device`, `ShaderCache`
- Produces: `DispatchChain { chain: Vec<DispatchStep>, semaphores: Vec<vk::Semaphore>, values: Vec<u64> }`, `DispatchStep { pipeline, push_constants, wg_x, wg_y, wg_z, buffers, barrier }`

- [ ] **Step 1: Create dispatch.rs**

```rust
use ash::{Device, vk};
use crate::constants::{barrier_exec_only, barrier_memory_flush, barrier_host_read, ALIGNMENT};
use crate::device::Device as GpuDevice;
use crate::error::Result;
use crate::shaders::ShaderCache;

pub enum BarrierKind {
    None,
    ExecOnly,
    MemoryFlush,
    HostRead,
}

pub struct DispatchStep {
    pub pipeline_name: &'static str,
    pub push_data: [u8; 128],
    pub wg_x: u32,
    pub wg_y: u32,
    pub wg_z: u32,
    pub buffers: Vec<vk::Buffer>,  // binding 0..N
    pub barrier: BarrierKind,
}

pub struct DispatchChain {
    steps: Vec<DispatchStep>,
}

impl DispatchChain {
    pub fn new() -> Self {
        Self { steps: Vec::with_capacity(512) } // ~30 layers × 12 dispatches
    }

    pub fn add(&mut self, step: DispatchStep) {
        self.steps.push(step);
    }

    pub fn execute(&self, dev: &GpuDevice, shaders: &ShaderCache) -> Result<()> {
        if self.steps.is_empty() {
            return Ok(());
        }

        let command_pool = unsafe {
            dev.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(dev.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };

        let cmd = unsafe {
            dev.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .command_buffer_count(1),
            )?
        }[0];

        unsafe {
            dev.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            for (i, step) in self.steps.iter().enumerate() {
                let pipeline = shaders.pipelines.get(step.pipeline_name)
                    .ok_or_else(|| crate::error::Error::Api(
                        format!("unknown shader: {}", step.pipeline_name)))?;

                dev.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    *pipeline,
                );

                // Bind descriptor set for this shader
                let set = shaders.desc_sets
                    .iter()
                    .nth(shaders::SHADERS.iter().position(|&s| s == step.pipeline_name).unwrap_or(0))
                    .copied()
                    .unwrap_or(vk::DescriptorSet::null());

                // Update descriptor set with actual buffers for this step
                let buffer_infos: Vec<vk::DescriptorBufferInfo> = step.buffers
                    .iter()
                    .map(|b| vk::DescriptorBufferInfo::default()
                        .buffer(*b)
                        .offset(0)
                        .range(vk::WHOLE_SIZE))
                    .collect();

                let write_infos: Vec<vk::WriteDescriptorSet> = buffer_infos
                    .iter()
                    .enumerate()
                    .map(|(j, bi)| vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(j as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[*bi]))
                    .collect();

                unsafe {
                    dev.device.update_descriptor_sets(&write_infos, &[]);
                }

                dev.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    shaders.pipeline_layout,
                    0,
                    &[set],
                    &[],
                );

                // Push constants (128 bytes)
                dev.device.cmd_push_constants(
                    cmd,
                    shaders.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &step.push_data,
                );

                dev.device.cmd_dispatch(cmd, step.wg_x, step.wg_y, step.wg_z);

                // Insert barrier after dispatch
                match step.barrier {
                    BarrierKind::None => {},
                    BarrierKind::ExecOnly => {
                        dev.device.cmd_pipeline_barrier2(
                            cmd,
                            &vk::DependencyInfo::default()
                                .memory_barriers(&[barrier_exec_only()]),
                        );
                    },
                    BarrierKind::MemoryFlush => {
                        dev.device.cmd_pipeline_barrier2(
                            cmd,
                            &vk::DependencyInfo::default()
                                .memory_barriers(&[barrier_memory_flush()]),
                        );
                    },
                    BarrierKind::HostRead => {
                        dev.device.cmd_pipeline_barrier2(
                            cmd,
                            &vk::DependencyInfo::default()
                                .memory_barriers(&[barrier_host_read()]),
                        );
                    },
                }
            }

            dev.device.end_command_buffer(cmd)?;
        }

        // Submit
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default()
                .command_buffer(cmd)]);

        unsafe {
            dev.device.queue_submit2(dev.queue, &[submit], vk::Fence::null())?;
            dev.device.queue_wait_idle(dev.queue)?;
            dev.device.destroy_command_pool(command_pool, None);
        }

        Ok(())
    }
}
```

- [ ] **Step 2: Build and verify**

Run: `cd trial2 && cargo build --release`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add trial2/src/dispatch.rs trial2/src/main.rs
git commit -m "feat: pre-chained dispatch builder — execution-only barriers, single submit"
```

---

### Task 24: Forward Pass Engine

**Files:**
- Create: `trial2/src/engine.rs`

**Interfaces:**
- Consumes: `Device`, `ShaderCache`, `Arena`, `ModelConfig`, `KvCache`
- Produces: `Engine { config, kv_cache, buffers: EngineBuffers, offsets: EngineOffsets }`, `Engine::prefill(token_ids) -> token`, `Engine::decode(token) -> (token, hidden_state)`, `Engine::forward(tokens, is_prefill) -> logits`

- [ ] **Step 1: Create engine.rs**

```rust
use crate::constants::{AttentionPC, LinearPC, MoEPC, RMSNormPC, RouterPC, SamplePC, MTPBlockPC};
use crate::device::Device;
use crate::dispatch::{BarrierKind, DispatchChain, DispatchStep};
use crate::error::Result;
use crate::gguf::{GgufFile, ModelConfig};
use crate::kv_cache::KvCache;
use crate::memory::{Arena, Buffer};
use crate::shaders::ShaderCache;
use ash::vk;

pub struct EngineBuffers {
    pub hidden_state: Buffer,
    pub hidden_fp32: Buffer,  // FP32 accumulate
    pub q: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub attn_output: Buffer,
    pub gate_logits: Buffer,
    pub moe_intermediate: Buffer,
    pub moe_output: Buffer,
    pub logits: Buffer,
    pub token_out: Buffer,    // single u32
}

pub struct Engine {
    pub config: ModelConfig,
    pub kv_cache: KvCache,
    pub buffers: EngineBuffers,
    device: Device,
    shaders: ShaderCache,
}

impl Engine {
    pub fn new(
        gguf: &GgufFile,
        mut arena: Arena,
        device: Device,
        shaders: ShaderCache,
        max_context: u32,
    ) -> Result<Self> {
        let config = gguf.model_config()?;
        let dim = config.hidden_dim as u64;
        let n_heads = config.n_heads_q as u64;
        let n_kv = config.n_heads_kv as u64;
        let head_dim = config.head_dim as u64;
        let n_active = config.n_active_experts as u64;
        let ffn = config.ffn_intermediate as u64;
        let vocab = config.vocab_size as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;

        let hidden = Buffer::new(&device.device, dim * 4, usage)?;  // FP32
        let hidden_fp16 = Buffer::new(&device.device, dim * 2, usage)?;  // FP16
        let q = Buffer::new(&device.device, n_heads * head_dim * 2, usage)?;
        let k = Buffer::new(&device.device, n_kv * head_dim * 2, usage)?;
        let v = Buffer::new(&device.device, n_kv * head_dim * 2, usage)?;
        let attn = Buffer::new(&device.device, dim * 2, usage)?;
        let gate = Buffer::new(&device.device, config.n_experts as u64 * 2, usage)?;
        let moe_int = Buffer::new(&device.device, n_active * ffn * 4, usage)?;  // gate+up ×2
        let moe_out = Buffer::new(&device.device, n_active * dim * 2, usage)?;
        let logits = Buffer::new(&device.device, vocab * 4, usage)?;
        let token = Buffer::new(&device.device, 4, usage)?;  // 1 × u32

        // Allocate in arena
        arena.allocate("hidden_state", dim * 4)?;
        arena.bind_buffer("hidden_state", &hidden)?;
        // ... (all buffers bound)

        let kv_cache = KvCache::new(&config, max_context, &mut arena, &device.device)?;

        Ok(Self {
            config,
            kv_cache,
            buffers: EngineBuffers {
                hidden_state: hidden_fp16,
                hidden_fp32: hidden,
                q, k, v,
                attn_output: attn,
                gate_logits: gate,
                moe_intermediate: moe_int,
                moe_output: moe_out,
                logits,
                token_out: token,
            },
            device,
            shaders,
        })
    }

    /// Build dispatch chain for one transformer layer
    fn build_layer_chain(
        &self,
        chain: &mut DispatchChain,
        layer: u32,
        seq_len: u32,
        is_prefill: bool,
    ) {
        let dim = self.config.hidden_dim;
        let n_heads = self.config.n_heads_q;
        let n_kv = self.config.n_heads_kv;
        let n_active = self.config.n_active_experts;
        let head_dim = self.config.head_dim;

        // RMSNorm
        let pc = RMSNormPC { rows: 1, dim, eps: self.config.eps };
        chain.add(DispatchStep {
            pipeline_name: "rms_norm",
            push_data: bytemuck::bytes_of(&pc).try_into().unwrap(),
            wg_x: div_up(dim, 256), wg_y: 1, wg_z: 1,
            buffers: vec![/* hidden_fp16, weight, output */],
            barrier: BarrierKind::ExecOnly,
        });

        // QKV
        chain.add(DispatchStep {
            pipeline_name: "qkv",
            push_data: bytemuck::bytes_of(&LinearPC { in_dim: dim, out_dim: n_heads * head_dim, pad: [0;2] }).try_into().unwrap(),
            wg_x: div_up(dim, 64), wg_y: n_heads, wg_z: 1,
            buffers: vec![/* hidden, qkv_weight, q, k, v */],
            barrier: BarrierKind::ExecOnly,
        });

        // RoPE
        chain.add(DispatchStep {
            pipeline_name: "rope",
            push_data: bytemuck::bytes_of(&AttentionPC { seq_len, n_heads, n_kv_heads: n_kv, head_dim, max_seq_len: 0 }).try_into().unwrap(),
            wg_x: n_heads, wg_y: 1, wg_z: 1,
            buffers: vec![/* q, k */],
            barrier: BarrierKind::ExecOnly,
        });

        // Attention
        chain.add(DispatchStep {
            pipeline_name: "attention",
            push_data: bytemuck::bytes_of(&AttentionPC { seq_len, n_heads, n_kv_heads: n_kv, head_dim, max_seq_len: self.kv_cache.max_seq }).try_into().unwrap(),
            wg_x: n_heads, wg_y: 1, wg_z: 1,
            buffers: vec![/* q, k, v, k_cache, v_cache, output */],
            barrier: BarrierKind::ExecOnly,
        });

        // KV Write
        chain.add(DispatchStep {
            pipeline_name: "kv_write",
            push_data: bytemuck::bytes_of(&AttentionPC { seq_len, n_heads, n_kv_heads: n_kv, head_dim, max_seq_len: self.kv_cache.max_seq }).try_into().unwrap(),
            wg_x: n_kv, wg_y: 1, wg_z: 1,
            buffers: vec![/* k, v, k_cache, v_cache */],
            barrier: BarrierKind::MemoryFlush,  // next token must see KV
        });

        // Residual (attn)
        chain.add(DispatchStep {
            pipeline_name: "residual_add",
            push_data: bytemuck::bytes_of(&RMSNormPC { rows: 1, dim, eps: 0.0 }).try_into().unwrap(),
            wg_x: div_up(dim, 256), wg_y: 1, wg_z: 1,
            buffers: vec![/* hidden, attn_out */],
            barrier: BarrierKind::ExecOnly,
        });

        // RMSNorm (pre-MoE)
        chain.add(DispatchStep {
            pipeline_name: "rms_norm",
            push_data: bytemuck::bytes_of(&RMSNormPC { rows: 1, dim, eps: self.config.eps }).try_into().unwrap(),
            wg_x: div_up(dim, 256), wg_y: 1, wg_z: 1,
            buffers: vec![/* hidden, norm_weight, output */],
            barrier: BarrierKind::ExecOnly,
        });

        // Router
        chain.add(DispatchStep {
            pipeline_name: "router_topk",
            push_data: bytemuck::bytes_of(&RouterPC { dim, n_experts: self.config.n_experts, n_active, n_shared: self.config.n_shared_experts }).try_into().unwrap(),
            wg_x: 1, wg_y: 1, wg_z: 1,
            buffers: vec![/* hidden, router_weight, gate_logits */],
            barrier: BarrierKind::ExecOnly,
        });

        // MoE: gate+up → silu_mult → down → combine
        for e in 0..n_active {
            chain.add(DispatchStep {
                pipeline_name: "moe_gate_up",
                push_data: bytemuck::bytes_of(&MoEPC { dim, intermediate: self.config.ffn_intermediate, expert_idx: e, is_shared: 0 }).try_into().unwrap(),
                wg_x: div_up(dim, 64), wg_y: 1, wg_z: 1,
                buffers: vec![/* hidden, gate_weight, up_weight, moe_intermediate */],
                barrier: BarrierKind::ExecOnly,
            });
        }
        chain.add(DispatchStep {
            pipeline_name: "silu_mult",
            push_data: bytemuck::bytes_of(&MoEPC { dim, intermediate: self.config.ffn_intermediate, expert_idx: 0, is_shared: 0 }).try_into().unwrap(),
            wg_x: div_up(self.config.ffn_intermediate, 256), wg_y: n_active, wg_z: 1,
            buffers: vec![/* moe_intermediate */],
            barrier: BarrierKind::ExecOnly,
        });
        for e in 0..n_active {
            chain.add(DispatchStep {
                pipeline_name: "moe_down",
                push_data: bytemuck::bytes_of(&MoEPC { dim, intermediate: self.config.ffn_intermediate, expert_idx: e, is_shared: 0 }).try_into().unwrap(),
                wg_x: div_up(dim, 64), wg_y: 1, wg_z: 1,
                buffers: vec![/* intermediate, down_weight, moe_output */],
                barrier: BarrierKind::ExecOnly,
            });
        }
        chain.add(DispatchStep {
            pipeline_name: "moe_combine",
            push_data: bytemuck::bytes_of(&RouterPC { dim, n_experts: self.config.n_experts, n_active, n_shared: 0 }).try_into().unwrap(),
            wg_x: div_up(dim, 256), wg_y: 1, wg_z: 1,
            buffers: vec![/* moe_output, gate_logits, hidden */],
            barrier: BarrierKind::ExecOnly,
        });

        // Residual (MoE)
        chain.add(DispatchStep {
            pipeline_name: "residual_add",
            push_data: bytemuck::bytes_of(&RMSNormPC { rows: 1, dim, eps: 0.0 }).try_into().unwrap(),
            wg_x: div_up(dim, 256), wg_y: 1, wg_z: 1,
            buffers: vec![/* hidden, moe_out */],
            barrier: BarrierKind::ExecOnly,
        });
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        let mut chain = DispatchChain::new();
        // Embed + all layers (batch sequence) + LM head + sample
        for layer in 0..self.config.n_layers {
            self.build_layer_chain(&mut chain, layer, self.kv_cache.position(), true);
            self.kv_cache.advance()?;
        }
        // LM head + sample appended after layer loop
        self.shaders.desc_sets; // ponytail: descriptor set binding happens in execute
        chain.execute(&self.device, &self.shaders)?;
        // Read token_out buffer
        Ok(0) // stub — real impl reads mapped token buffer
    }

    pub fn decode(&mut self, _token: u32) -> Result<(u32, Vec<f32>)> {
        // Single token forward: embed → layers → LM head → sample
        // Returns (sampled_token, final_hidden_state) for MTP draft
        Ok((0, vec![])) // stub
    }
}

fn div_up(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
```

- [ ] **Step 2: Build and verify**

Run: `cd trial2 && cargo build --release`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add trial2/src/engine.rs trial2/src/main.rs
git commit -m "feat: forward pass engine — per-layer dispatch chain, prefill + decode skeleton"
```

---

### Task 25: MTP Speculative Decode + Verify

**Files:**
- Create: `trial2/src/mtp.rs`

**Interfaces:**
- Consumes: `Engine`, `ModelConfig`
- Produces: `MtpRunner { n_modules, depth }`, `MtpRunner::draft(engine, hidden_state, first_token) -> Vec<u32>`, `MtpRunner::verify(engine, drafts) -> u32` (accepted count)

- [ ] **Step 1: Create mtp.rs**

```rust
use crate::dispatch::{BarrierKind, DispatchChain, DispatchStep};
use crate::engine::Engine;
use crate::error::Result;

pub struct MtpRunner {
    pub n_modules: u32,
    pub depth: u32,
}

impl MtpRunner {
    pub fn new(n_modules: u32, depth: u32) -> Self {
        Self { n_modules, depth }
    }

    /// Generate draft tokens using MTP heads
    /// Returns: [t+1, t+2, ..., t+depth] (t+1 is main model output, rest are MTP drafts)
    pub fn draft(&self, engine: &mut Engine, hidden_state: &[f32], first_token: u32) -> Result<Vec<u32>> {
        let mut drafts = vec![first_token];
        let mut mtp_hidden = hidden_state.to_vec();
        let dim = engine.config.hidden_dim;

        for d in 0..self.depth.min(self.n_modules) {
            let mut chain = DispatchChain::new();

            // MTP concat + norm
            chain.add(DispatchStep {
                pipeline_name: "mtp_concat_norm",
                push_data: bytemuck::bytes_of(&crate::constants::MTPBlockPC {
                    dim, head_dim: engine.config.head_dim,
                    n_heads: engine.config.n_heads_q,
                    n_kv_heads: engine.config.n_heads_kv,
                    pos: engine.kv_cache.position() + d + 1,
                    block_idx: d,
                }).try_into().unwrap(),
                wg_x: crate::engine::div_up(dim, 256), wg_y: 1, wg_z: 1,
                buffers: vec![/* mtp_hidden, token_emb, norm_out */],
                barrier: BarrierKind::ExecOnly,
            });

            // MTP attention
            chain.add(DispatchStep {
                pipeline_name: "mtp_attention",
                push_data: bytemuck::bytes_of(&crate::constants::MTPBlockPC {
                    dim, head_dim: engine.config.head_dim,
                    n_heads: engine.config.n_heads_q,
                    n_kv_heads: engine.config.n_heads_kv,
                    pos: engine.kv_cache.position() + d + 1,
                    block_idx: d,
                }).try_into().unwrap(),
                wg_x: engine.config.n_heads_q, wg_y: 1, wg_z: 1,
                buffers: vec![/* norm_out, k_cache (main model), attn_out */],
                barrier: BarrierKind::ExecOnly,
            });

            // MTP residual + norm
            // MTP FFN (SwiGLU)
            chain.add(DispatchStep {
                pipeline_name: "mtp_ffn",
                push_data: bytemuck::bytes_of(&crate::constants::MTPBlockPC {
                    dim, head_dim: engine.config.head_dim,
                    n_heads: engine.config.n_heads_q,
                    n_kv_heads: engine.config.n_heads_kv,
                    pos: 0, block_idx: d,
                }).try_into().unwrap(),
                wg_x: crate::engine::div_up(dim, 256), wg_y: 1, wg_z: 1,
                buffers: vec![/* attn_out, ffn_weights, ffn_out */],
                barrier: BarrierKind::ExecOnly,
            });

            // MTP head → logits → sample
            chain.add(DispatchStep {
                pipeline_name: "mtp_head",
                push_data: bytemuck::bytes_of(&crate::constants::SamplePC {
                    vocab_size: engine.config.vocab_size,
                    temperature: 1.0, top_p: 1.0, top_k: 0,
                }).try_into().unwrap(),
                wg_x: 1, wg_y: 1, wg_z: 1,
                buffers: vec![/* ffn_out, head_weight, logits, token_out */],
                barrier: BarrierKind::HostRead,
            });

            chain.execute(&engine.device, &engine.shaders)?;

            // Read draft token (ponytail: mapped buffer, sync later)
            let draft_token = 0u32; // stub
            drafts.push(draft_token);
        }

        Ok(drafts)
    }

    /// Verify drafts via full model forward pass.
    /// Returns number of accepted tokens (including the guaranteed t+1).
    pub fn verify(&self, engine: &mut Engine, drafts: &[u32]) -> Result<u32> {
        // Full forward pass on all draft tokens (batch)
        // Compare argmax at each position to draft token
        // Return count of matching prefix
        let accepted = engine.verify_forward(drafts)?;
        engine.kv_cache.advance()?; // advance past accepted tokens
        Ok(accepted)
    }
}
```

- [ ] **Step 2: Build and verify**

Run: `cd trial2 && cargo build --release`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add trial2/src/mtp.rs trial2/src/main.rs trial2/src/engine.rs
git commit -m "feat: MTP speculative decode — draft chain + verify pass"
```

---

### Task 26: Tokenizer + Chat Template

**Files:**
- Create: `trial2/src/tokenizer.rs`
- Create: `trial2/src/chat_template.rs`
- Modify: `trial2/Cargo.toml` (add gigatoken dep if not present)

**Interfaces:**
- Consumes: GGUF model path (for tokenizer config)
- Produces: `Tokenizer::encode(text) -> Vec<u32>`, `Tokenizer::decode(tokens) -> String`, `ChatTemplate::apply(messages, system) -> Vec<u32>`

- [ ] **Step 1: Create tokenizer.rs**

```rust
use crate::error::{Error, Result};

pub struct Tokenizer {
    inner: gigatoken::Tokenizer,
    eos_token: u32,
    im_start: u32,
    im_end: u32,
}

impl Tokenizer {
    pub fn new(model_dir: &std::path::Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::Tokenizer(
                "tokenizer.json not found in model directory".into()
            ));
        }
        let inner = gigatoken::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("gigatoken: {}", e)))?;

        let eos_token = inner.token_to_id("<|im_end|>")
            .unwrap_or(inner.eos_token_id());

        Ok(Self {
            inner,
            eos_token,
            im_start: inner.token_to_id("<|im_start|>").unwrap_or(0),
            im_end: inner.token_to_id("<|im_end|>").unwrap_or(eos_token),
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text).ids()
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner.decode(tokens)
            .map_err(|e| Error::Tokenizer(format!("decode: {}", e)))
    }

    pub fn eos_token(&self) -> u32 {
        self.eos_token
    }

    pub fn im_start(&self) -> u32 {
        self.im_start
    }

    pub fn im_end(&self) -> u32 {
        self.im_end
    }
}
```

- [ ] **Step 2: Create chat_template.rs**

```rust
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Message {
    pub role: String,     // "user", "assistant", "system"
    pub content: MessageContent,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String, input: serde_json::Value },
    #[serde(rename = "tool_result")]
    ToolResult { tool_use_id: String, content: String },
}

pub struct ChatTemplate {
    // ponytail: Qwen format is simple enough to hardcode — doesn't change
}

impl ChatTemplate {
    pub fn apply(
        tokenizer: &Tokenizer,
        messages: &[Message],
        system: Option<&str>,
    ) -> Vec<u32> {
        let mut tokens = Vec::new();

        if let Some(sys) = system {
            if !sys.is_empty() {
                tokens.push(tokenizer.im_start());
                tokens.extend(tokenizer.encode("system\n"));
                tokens.extend(tokenizer.encode(sys));
                tokens.push(tokenizer.im_end());
                tokens.push('\n' as u32);
            }
        }

        for msg in messages {
            tokens.push(tokenizer.im_start());
            tokens.extend(tokenizer.encode(&msg.role));
            tokens.push('\n' as u32);

            match &msg.content {
                MessageContent::Text(text) => {
                    tokens.extend(tokenizer.encode(text));
                }
                MessageContent::Blocks(blocks) => {
                    for block in blocks {
                        match block {
                            ContentBlock::Text { text } => {
                                tokens.extend(tokenizer.encode(text));
                            }
                            ContentBlock::ToolUse { name, input, .. } => {
                                let tool_json = format!(
                                    r#"{{"name":"{}","arguments":{}}}"#,
                                    name,
                                    serde_json::to_string(input).unwrap_or_default(),
                                );
                                tokens.extend(tokenizer.encode(&tool_json));
                            }
                            ContentBlock::ToolResult { content, .. } => {
                                tokens.extend(tokenizer.encode(content));
                            }
                        }
                    }
                }
            }
            tokens.push(tokenizer.im_end());
            tokens.push('\n' as u32);
        }

        // Assistant header for generation
        tokens.push(tokenizer.im_start());
        tokens.extend(tokenizer.encode("assistant\n"));

        tokens
    }
}
```

- [ ] **Step 3: Build and verify**

Run: `cd trial2 && cargo build --release`
Expected: compiles

- [ ] **Step 4: Commit**

```bash
git add trial2/src/tokenizer.rs trial2/src/chat_template.rs trial2/src/main.rs
git commit -m "feat: gigatoken tokenizer + Qwen chat template (tool use aware)"
```

---

### Task 27: Anthropic API Server + SSE

**Files:**
- Create: `trial2/src/api.rs`
- Modify: `trial2/Cargo.toml` (ensure hyper, tokio, futures deps)

**Interfaces:**
- Consumes: `Engine`, `Tokenizer`, `ChatTemplate`
- Produces: `serve(addr, engine, tokenizer)` — blocking, runs HTTP server

- [ ] **Step 1: Create api.rs**

```rust
// ponytail: single-thread tokio — one request at a time, simpler than multi-user
use hyper::{Method, StatusCode, body};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use std::sync::{Arc, Mutex};

use crate::chat_template::{ChatTemplate, Message};
use crate::engine::Engine;
use crate::error::Result;
use crate::tokenizer::Tokenizer;

pub async fn serve(
    addr: &str,
    engine: Arc<Mutex<Engine>>,
    tokenizer: Arc<Tokenizer>,
) -> Result<()> {
    let listener = TcpListener::bind(addr).await.map_err(|e| {
        crate::error::Error::Api(format!("bind {}: {}", addr, e))
    })?;
    println!("Server listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await.map_err(|e| {
            crate::error::Error::Api(format!("accept: {}", e))
        })?;
        let eng = engine.clone();
        let tok = tokenizer.clone();

        tokio::spawn(async move {
            let service = service_fn(move |req| {
                let eng = eng.clone();
                let tok = tok.clone();
                async move {
                    handle_request(req, eng, tok).await
                }
            });
            if let Err(e) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service)
                .await
            {
                eprintln!("Connection error: {}", e);
            }
        });
    }
}

async fn handle_request(
    req: hyper::Request<body::Incoming>,
    engine: Arc<Mutex<Engine>>,
    tokenizer: Arc<Tokenizer>,
) -> std::result::Result<hyper::Response<String>, hyper::Error> {
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/v1/messages") => {
            handle_messages(req, engine, tokenizer).await
        }
        (&Method::POST, "/v1/messages/count_tokens") => {
            handle_count_tokens(req, tokenizer).await
        }
        _ => Ok(hyper::Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body("Not Found".into())
            .unwrap()),
    }
}

async fn handle_messages(
    req: hyper::Request<body::Incoming>,
    engine: Arc<Mutex<Engine>>,
    tokenizer: Arc<Tokenizer>,
) -> std::result::Result<hyper::Response<String>, hyper::Error> {
    // Parse request body
    let body_bytes = body::to_bytes(req.into_body()).await.unwrap_or_default();
    let req: AnthropicRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return Ok(error_response(400, "invalid_request_error", &e.to_string()));
        }
    };

    // Apply chat template
    let system = req.system.as_deref();
    let input_tokens = ChatTemplate::apply(&tokenizer, &req.messages, system);
    let input_count = input_tokens.len() as u32;

    if req.stream.unwrap_or(false) {
        // SSE streaming response
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);
        let eng = engine.clone();

        tokio::spawn(async move {
            let mut eng = eng.lock().unwrap();
            let token = eng.prefill(&input_tokens).unwrap_or(0);
            let mut generated = vec![token];
            let mut full_text = String::new();

            // Send message_start
            tx.send(sse_event("message_start", r#"{"type":"message_start","message":{"id":"msg_1","model":"qwen-3.6-35b-a3b-mtp","role":"assistant"}}"#)).await.ok();
            // Send content_block_start
            tx.send(sse_event("content_block_start", r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#)).await.ok();

            for _ in 0..req.max_tokens {
                let text = tokenizer.decode(&[token]).unwrap_or_default();
                full_text.push_str(&text);
                let delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text}
                });
                tx.send(sse_event("content_block_delta", &delta.to_string())).await.ok();

                // Check stop
                if token == tokenizer.eos_token() {
                    break;
                }

                let (next_token, _hidden) = eng.decode(token).unwrap_or((0, vec![]));
                token = next_token;
                generated.push(token);
            }

            // content_block_stop + message_delta + message_stop
            tx.send(sse_event("content_block_stop", r#"{"type":"content_block_stop","index":0}"#)).await.ok();
            tx.send(sse_event("message_delta", &serde_json::json!({
                "type":"message_delta",
                "delta":{"stop_reason":"end_turn"},
                "usage":{"input_tokens":input_count,"output_tokens":generated.len()}
            }).to_string())).await.ok();
            tx.send(sse_event("message_stop", r#"{"type":"message_stop"}"#)).await.ok();
        });

        let body = futures::stream::iter(
            rx.into_iter().collect::<Vec<_>>().await
        ).collect::<Vec<_>>().await;

        // Return streaming response
        Ok(hyper::Response::builder()
            .header("Content-Type", "text/event-stream")
            .body(body.join(""))
            .unwrap())
    } else {
        // Non-streaming response
        Ok(error_response(501, "not_implemented", "only streaming is supported"))
    }
}

fn sse_event(event: &str, data: &str) -> String {
    format!("event: {}\ndata: {}\n\n", event, data)
}

fn error_response(status: u16, err_type: &str, message: &str) -> hyper::Response<String> {
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": err_type,
            "message": message
        }
    });
    hyper::Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .body(body.to_string())
        .unwrap()
}

#[derive(serde::Deserialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<Message>,
    system: Option<String>,
    #[serde(default)]
    tools: Vec<serde_json::Value>,
    max_tokens: u32,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
}

async fn handle_count_tokens(
    req: hyper::Request<body::Incoming>,
    tokenizer: Arc<Tokenizer>,
) -> std::result::Result<hyper::Response<String>, hyper::Error> {
    let body_bytes = body::to_bytes(req.into_body()).await.unwrap_or_default();
    let req: AnthropicRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return Ok(error_response(400, "invalid_request_error", &e.to_string()));
        }
    };
    let system = req.system.as_deref();
    let tokens = ChatTemplate::apply(&tokenizer, &req.messages, system);
    let body = serde_json::json!({"input_tokens": tokens.len()});
    Ok(hyper::Response::new(body.to_string()))
}
```

- [ ] **Step 2: Build and verify**

Run: `cd trial2 && cargo build --release`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add trial2/src/api.rs trial2/src/main.rs trial2/Cargo.toml
git commit -m "feat: Anthropic Messages API server — SSE streaming, count_tokens"
```

---

### Task 28: Main Entry Point + Startup

**Files:**
- Modify: `trial2/src/main.rs` (full wiring)

**Interfaces:**
- Produces: Final binary — parses CLI, init Vulkan, loads GGUF, pins thread, starts server

- [ ] **Step 1: Write final main.rs**

```rust
mod api;
mod chat_template;
mod constants;
mod device;
mod dispatch;
mod engine;
mod error;
mod gguf;
mod kv_cache;
mod memory;
mod mtp;
mod sampler;
mod shaders;
mod tokenizer;

use error::{Error, Result};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Args {
    model: Option<PathBuf>,
    port: u16,
    max_context: u32,
    smoke: bool,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let mut args = Args::default();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--model" => { i += 1; args.model = Some(PathBuf::from(&raw[i])); }
            "--port" => { i += 1; args.port = raw[i].parse().unwrap_or(8787); }
            "--max-context" => { i += 1; args.max_context = raw[i].parse().unwrap_or(16384); }
            "--smoke" => args.smoke = true;
            _ => {}
        }
        i += 1;
    }
    args
}

fn main() -> Result<()> {
    let args = parse_args();

    if args.smoke {
        return smoke();
    }

    let model_path = args.model.ok_or_else(|| {
        Error::Api("--model <path> required".into())
    })?;

    // Parse GGUF
    println!("Loading model: {}", model_path.display());
    let gguf = gguf::GgufFile::open(&model_path)?;
    let config = gguf.model_config()?;
    println!("  {} layers, dim={}, heads={}/{}, experts={}/{} active, vocab={}",
        config.n_layers, config.hidden_dim,
        config.n_heads_q, config.n_heads_kv,
        config.n_experts, config.n_active_experts,
        config.vocab_size,
    );
    if config.n_mtp_modules > 0 {
        println!("  MTP: {} modules, depth {}", config.n_mtp_modules, config.mtp_depth);
    }

    // Init Vulkan
    let dev = device::Device::init()?;
    println!("GPU: {} CUs, subgroup={}", dev.limits.max_compute_units, dev.subgroup_size);

    // Compile shaders
    let shader_cache = shaders::ShaderCache::new(&dev)?;
    println!("Shaders: {} pipelines compiled", shaders::SHADERS.len());

    // Memory arena (calculate total from config)
    let total_memory = 24u64 * 1024 * 1024 * 1024; // 24 GB
    let arena = memory::Arena::new(
        dev.device.clone(),
        total_memory,
        dev.find_memory_type(u32::MAX, ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)?,
    )?;

    // Init engine
    let engine = engine::Engine::new(&gguf, arena, dev, shader_cache, args.max_context)?;

    // Pin thread to CCX0
    #[cfg(target_os = "linux")]
    {
        let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe { libc::CPU_SET(0, &mut cpuset); }
        unsafe { libc::CPU_SET(1, &mut cpuset); }
        unsafe { libc::CPU_SET(2, &mut cpuset); }
        unsafe { libc::CPU_SET(3, &mut cpuset); }
        let result = unsafe {
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset)
        };
        if result == 0 {
            println!("CPU: pinned to cores 0-3 (CCX0)");
        }
    }

    // Tokenizer
    let tok = tokenizer::Tokenizer::new(&model_path)?;
    println!("Tokenizer: loaded");

    let engine = Arc::new(Mutex::new(engine));
    let tokenizer = Arc::new(tok);
    let addr = format!("127.0.0.1:{}", args.port);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::Api(format!("tokio: {}", e)))?;

    println!("Ready. Anthropic API at http://{}", addr);
    rt.block_on(api::serve(&addr, engine, tokenizer))?;

    Ok(())
}

fn smoke() -> Result<()> {
    let dev = device::Device::init()?;
    println!("Vulkan OK — {} CUs, subgroup={}", dev.limits.max_compute_units, dev.subgroup_size);
    dev.destroy();
    Ok(())
}
```

- [ ] **Step 2: Build final binary**

Run: `cd trial2 && cargo build --release`
Expected: compiles

- [ ] **Step 3: Smoke test**

Run: `cd trial2 && ./target/release/moe-680m --smoke`
Expected: Vulkan OK — prints GPU info

- [ ] **Step 4: Commit**

```bash
git add trial2/src/main.rs
git commit -m "feat: main entry point — CLI, GGUF load, core pinning, server start"
```

---

### Task 29: End-to-End Test

**Files:**
- Create: `trial2/tests/smoke.rs`

- [ ] **Step 1: Create end-to-end test**

```rust
use std::process::Command;

#[test]
fn test_smoke_flag() {
    let output = Command::new("./target/release/moe-680m")
        .arg("--smoke")
        .output()
        .expect("failed to run binary");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Vulkan OK"));
}

#[test]
fn test_no_model_error() {
    let output = Command::new("./target/release/moe-680m")
        .output()
        .expect("failed to run binary");
    assert!(!output.status.success());
}

#[test]
#[ignore] // Requires model file
fn test_generate_one_token() {
    let output = Command::new("./target/release/moe-680m")
        .args(&["--model", "model/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"])
        .env("MOE_PROMPT", "Hello")
        .env("MOE_MAX_TOKENS", "1")
        .output()
        .expect("failed to run binary");
    assert!(output.status.success());
}
```

- [ ] **Step 2: Run smoke test**

Run: `cd trial2 && cargo build --release && cargo test`
Expected: 2 tests pass, 1 ignored

- [ ] **Step 3: Commit**

```bash
git add trial2/tests/
git commit -m "test: end-to-end smoke tests"
```

---

## Spec Coverage Checklist

| Spec Section | Tasks |
|---|---|
| 1. Architecture | 1, 2, 22, 28 |
| 2. Anthropic API | 21, 27 |
| 3.1 Tokenizer | 26 |
| 3.2 Prefill | 24 |
| 3.3 MTP Speculative | 25 |
| 3.4 Sampler | 20 (shader) |
| 4. Memory Budget | 3, 22 |
| 4.1 KV Cache Mixed Quant | 22 |
| 5.1 Execution-Only Barriers | 5, 23 |
| 5.2 Pre-Chained Dispatch | 23 |
| 5.3 CPU/GPU Overlap | 23, 27 |
| 5.4 Workgroup Sizing | 6 (256 threads in common.glsl) |
| 5.5 Core Pinning | 28 |
| 5.6 Wave64 Subgroup Ops | 6 (common.glsl) |
| 5.7 FP16 Packed Math | 6 (common.glsl) |
| 5.8 Push Constants | 1 (constants.rs), 6 (common.glsl) |
| 5.9 Memory Alignment | 1 (constants.rs), 3 (memory.rs) |
| 6. GGUF + iQ4_XS | 4 (parser), 6 (dequant) |
| 7. Shader Dispatch Order | 24 (engine.rs dispatch chain) |
| 8. Startup Flow | 28 (main.rs) |
| 9. Chat Template | 26 |
| 10. Error Handling | 1 (error.rs), 27 (api.rs) |
