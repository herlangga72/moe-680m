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
mod shaders;
mod tokenizer;
mod weights;

use error::{Error, Result};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Extract the physical device name string from Vulkan properties.
fn device_name(instance: &ash::Instance, physical: ash::vk::PhysicalDevice) -> String {
    let props = unsafe { instance.get_physical_device_properties(physical) };
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            props.device_name.as_ptr() as *const u8,
            props.device_name.len(),
        )
    };
    std::str::from_utf8(bytes)
        .unwrap_or("unknown")
        .trim_end_matches('\0')
        .to_string()
}

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

struct Args {
    model: Option<PathBuf>,
    port: u16,
    max_context: u32,
    smoke: bool,
    test_forward: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self { model: None, port: 8787, max_context: 4096, smoke: false, test_forward: false }
    }
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let mut args = Args::default();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--model" => {
                i += 1;
                args.model = Some(PathBuf::from(&raw[i]));
            }
            "--port" => {
                i += 1;
                args.port = raw[i].parse().unwrap_or(8787);
            }
            "--max-context" => {
                i += 1;
                args.max_context = raw[i].parse().unwrap_or(4096);
            }
            "--smoke" => args.smoke = true,
            "--test-forward" => args.test_forward = true,
            _ => {}
        }
        i += 1;
    }
    args
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = parse_args();

    // --smoke flag: quick GPU sanity check
    if args.smoke {
        return smoke();
    }

    // --model <path> is required for normal operation
    let model_path = args.model.ok_or_else(|| {
        Error::Api("--model <path> required (use --smoke for GPU check)".into())
    })?;

    // ---- 1. Load GGUF ----
    println!("Loading model: {}", model_path.display());
    let gguf = gguf::GgufFile::open(&model_path)?;
    let config = gguf.model_config()?;
    println!(
        "  {} layers, dim={}, heads={}/{}, experts={}/{} active, vocab={}",
        config.n_layers,
        config.hidden_dim,
        config.n_heads_q,
        config.n_heads_kv,
        config.n_experts,
        config.n_active_experts,
        config.vocab_size,
    );
    if config.n_mtp_modules > 0 {
        println!(
            "  MTP: {} modules, depth {}",
            config.n_mtp_modules, config.mtp_depth
        );
    }

    // ---- 2. Initialise Vulkan device ----
    let dev = device::Device::init()?;
    let gpu_name = device_name(&dev.instance, dev.physical);
    println!(
        "GPU: {} (subgroup={}, max_workgroup_invocations={})",
        gpu_name,
        dev.subgroup_size,
        dev.limits.max_compute_work_group_invocations,
    );

    // ---- 3. Compile shaders ----
    let shader_cache = shaders::ShaderCache::new(&dev)?;
    println!("Shaders: {} pipelines compiled", shaders::SHADERS.len());

    // ---- 4. Allocate GPU memory arena (24 GB) ----
    // UMA (iGPU): compute buffers (~1 GB) + KV cache
    let kv_per_token = config.n_heads_kv as u64 * (config.head_dim as u64 / 32 * 18 + config.head_dim as u64);
    let kv_total = config.n_layers as u64 * args.max_context as u64 * kv_per_token;
    let arena_size = (1u64 << 30) + kv_total; // 1 GB compute + KV
    eprintln!("Arena: {} MB (KV: {} MB for {} ctx)",
        arena_size >> 20, kv_total >> 20, args.max_context);
    let mem_flags = ash::vk::MemoryPropertyFlags::HOST_VISIBLE
        | ash::vk::MemoryPropertyFlags::HOST_COHERENT;
    let mem_type = dev.find_memory_type(u32::MAX, mem_flags)
        .or_else(|_| dev.find_memory_type(u32::MAX, ash::vk::MemoryPropertyFlags::HOST_VISIBLE))?;
    let arena = memory::Arena::new(dev.device.clone(), arena_size, mem_type)?;

    // ---- 5. Create inference engine ----
    let engine = engine::Engine::new(&gguf, arena, dev, shader_cache, args.max_context)?;

    if args.test_forward {
        return test_forward(engine, &gguf);
    }

    // ---- 6. Pin main thread to CCX0 (cores 0-3) ----
    #[cfg(target_os = "linux")]
    {
        let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe {
            libc::CPU_SET(0, &mut cpuset);
        }
        unsafe {
            libc::CPU_SET(1, &mut cpuset);
        }
        unsafe {
            libc::CPU_SET(2, &mut cpuset);
        }
        unsafe {
            libc::CPU_SET(3, &mut cpuset);
        }
        let result = unsafe {
            libc::sched_setaffinity(
                0,
                std::mem::size_of::<libc::cpu_set_t>(),
                &cpuset,
            )
        };
        if result == 0 {
            println!("CPU: pinned to cores 0-3 (CCX0)");
        }
    }

    // ---- 7. Load tokenizer ----
    let tok = tokenizer::Tokenizer::new(&model_path)?;
    println!("Tokenizer: loaded");

    // ---- 8. Start API server ----
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

// ---------------------------------------------------------------------------
// Smoke test — GPU initialisation only, no model required
// ---------------------------------------------------------------------------

fn pc_bytes<T: bytemuck::NoUninit>(pc: &T) -> [u8; 128] {
    let mut buf = [0u8; 128];
    let src = bytemuck::bytes_of(pc);
    let len = src.len().min(128);
    buf[..len].copy_from_slice(&src[..len]);
    buf
}

fn test_forward(mut engine: engine::Engine, gguf: &gguf::GgufFile) -> Result<()> {
    use ash::vk;
    println!("\n=== GPU forward-pass test ===");

    let dev = &engine.device;
    let dim = engine.config.hidden_dim;

    // Weight pool for layer 0
    let mut wp = weights::WeightPool::new(dev, 128 * 1024 * 1024)?; // 128 MB

    // Upload layer 0 weights
    let w_norm = wp.upload(gguf, "blk.0.attn_norm.weight")?;
    let w_qkv = wp.upload(gguf, "blk.0.attn_qkv.weight")?;
    println!("Uploaded: attn_norm ({}B) + attn_qkv ({}B)", w_norm.range, w_qkv.range);
    // Check if norm weights have 16-byte garbage prefix
    for name in ["output_norm.weight", "blk.0.attn_norm.weight", "blk.1.attn_norm.weight"] {
        if let Some(t) = gguf.find_tensor(name) {
            let raw = gguf.tensor_data(t);
            let f32s: &[f32] = bytemuck::cast_slice(&raw[..32.min(raw.len())]);
            println!("  {} (off={}): f32[0..4]={:?}", name, t.offset, &f32s[..4.min(f32s.len())]);
        }
    }

    // Allocate I/O buffers
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
    let io_size = (dim * 8) as u64; // FP32 input + FP16 hidden state
    let io_buf = memory::Buffer::new(&dev.device, io_size, usage)?;
    let q_buf = memory::Buffer::new(&dev.device, (16 * 256 * 4) as u64, usage)?; // q: FP32
    let k_buf = memory::Buffer::new(&dev.device, (2 * 256 * 4) as u64, usage)?; // k: FP32
    let v_buf = memory::Buffer::new(&dev.device, (2 * 256 * 4) as u64, usage)?;

    let io_mem_size = io_size + q_buf.size + k_buf.size + v_buf.size + 3 * 128;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(io_mem_size)
        .memory_type_index(dev.find_memory_type(u32::MAX,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?);
    let io_mem = unsafe { dev.device.allocate_memory(&alloc_info, None)? };
    let mut off = 0u64;
    for buf in [&io_buf, &q_buf, &k_buf, &v_buf] {
        unsafe { dev.device.bind_buffer_memory(buf.handle, io_mem, off)?; }
        off += (buf.size + 127) & !127;
    }

    // Fill input with 1.0 (hidden state)
    let ptr = unsafe { dev.device.map_memory(io_mem, 0, io_mem_size, vk::MemoryMapFlags::empty())? } as *mut f32;
    for i in 0..dim as usize { unsafe { *ptr.add(i) = 1.0f32; } }
    unsafe { dev.device.unmap_memory(io_mem); }

    // Test RMSNorm alone first
    let ptr = unsafe { dev.device.map_memory(io_mem, 0, io_mem_size, vk::MemoryMapFlags::empty())? } as *mut f32;
    for i in 0..dim as usize { unsafe { *ptr.add(i) = 1.0f32; } }
    unsafe { dev.device.unmap_memory(io_mem); }

    // Upload real FP32 norm weight
    let test_w = wp.upload(gguf, "blk.0.attn_norm.weight")?;

    let full = vk::DescriptorBufferInfo::default().buffer(io_buf.handle).offset(0).range(vk::WHOLE_SIZE);
    let out_bi = vk::DescriptorBufferInfo::default().buffer(io_buf.handle).offset((dim * 4) as u64).range(vk::WHOLE_SIZE);
    let q_bi = vk::DescriptorBufferInfo::default().buffer(q_buf.handle).offset(0).range(vk::WHOLE_SIZE);

    // RMSNorm only, then read output
    let mut chain = dispatch::DispatchChain::new();
    chain.add(dispatch::DispatchStep {
        pipeline_name: "rms_norm",
        push_data: pc_bytes(&constants::RMSNormPC { rows: 1, dim, eps: 1e-6 }),
        workgroup_x: (dim + 255) / 256, workgroup_y: 1, workgroup_z: 1,
        buffers: vec![full, test_w, test_w, out_bi, test_w],
        barrier: dispatch::BarrierKind::MemoryFlush,
    });

    // Pre-fill q_buf with sentinel 99.0 to detect shader write
    let q_off = (io_buf.size + 127) & !127;
    let ptr2 = unsafe { dev.device.map_memory(io_mem, q_off, q_buf.size, vk::MemoryMapFlags::empty())? } as *mut f32;
    unsafe { *ptr2 = 99.0f32; *ptr2.add(1) = 99.0f32; }
    unsafe { dev.device.unmap_memory(io_mem); }

    chain.execute(dev, &engine.shaders)?;

    let ptr3 = unsafe { dev.device.map_memory(io_mem, 0, io_buf.size, vk::MemoryMapFlags::empty())? } as *const f32;
    let rms_out = unsafe { std::slice::from_raw_parts(ptr3.add(dim as usize), 8) };
    println!("RMSNorm out[0..4] (real weight × 1.0) = {:?}", &rms_out[..4]);
    let ok = !rms_out[0].is_nan() && rms_out[0].abs() > 0.001;
    println!("RMSNorm: {} first={}", if ok { "PASS" } else { "FAIL" }, rms_out[0]);
    unsafe { dev.device.unmap_memory(io_mem); }
    unsafe { dev.device.unmap_memory(io_mem); }

    unsafe {
        dev.device.destroy_buffer(io_buf.handle, None);
        dev.device.destroy_buffer(q_buf.handle, None);
        dev.device.destroy_buffer(k_buf.handle, None);
        dev.device.destroy_buffer(v_buf.handle, None);
        dev.device.free_memory(io_mem, None);
    }
    wp.destroy();

    println!("GPU forward-pass: OK");
    Ok(())
}

fn smoke() -> Result<()> {
    let mut dev = device::Device::init()?;
    let gpu_name = device_name(&dev.instance, dev.physical);
    println!(
        "Vulkan OK — {} (subgroup={})",
        gpu_name,
        dev.subgroup_size,
    );
    dev.destroy();
    Ok(())
}
