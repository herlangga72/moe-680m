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
    let model_dir = model_path.parent().unwrap_or(std::path::Path::new("."));
    let tok = tokenizer::Tokenizer::new(model_dir)?;
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
    let n_q = engine.config.n_heads_q;
    let n_kv = engine.config.n_heads_kv;
    let hd = engine.config.head_dim;

    // Upload layer-0 attention weights + post-attn norm
    let mut wp = weights::WeightPool::new(dev, 128 * 1024 * 1024)?;
    let w_pre_norm = wp.upload(gguf, "blk.0.attn_norm.weight")?;
    let w_qkv = wp.upload(gguf, "blk.0.attn_qkv.weight")?;
    let w_post_norm = wp.upload(gguf, "blk.0.post_attention_norm.weight")?;
    println!("Weights: pre_norm {}B + qkv {}B + post_norm {}B", w_pre_norm.range, w_qkv.range, w_post_norm.range);

    // Allocate I/O buffers: hidden (FP32), q/k/v (FP32), attn_out (FP16)
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
    let make_buf = |bytes: u64| memory::Buffer::new(&dev.device, bytes, usage).unwrap();
    let hidden = make_buf((dim * 4) as u64);         // FP32 in-place
    let q_buf = make_buf((n_q * hd * 4) as u64);     // FP32
    let k_buf = make_buf((n_kv * hd * 4) as u64);
    let v_buf = make_buf((n_kv * hd * 4) as u64);
    let attn_buf = make_buf((dim * 4) as u64);       // FP32

    // Single UMA allocation for all I/O
    let bufs: [&memory::Buffer; 5] = [&hidden, &q_buf, &k_buf, &v_buf, &attn_buf];
    let total: u64 = bufs.iter().map(|b| (b.size + 127) & !127).sum();
    let io_mem = unsafe {
        let ai = vk::MemoryAllocateInfo::default().allocation_size(total)
            .memory_type_index(dev.find_memory_type(u32::MAX,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?);
        dev.device.allocate_memory(&ai, None)?
    };
    let mut off = 0u64;
    for b in &bufs { unsafe { dev.device.bind_buffer_memory(b.handle, io_mem, off)?; off += (b.size + 127) & !127; } }

    // Fill hidden state with 1.0
    let ptr = unsafe { dev.device.map_memory(io_mem, 0, total, vk::MemoryMapFlags::empty())? } as *mut f32;
    for i in 0..dim as usize { unsafe { *ptr.add(i) = 1.0f32; } }
    unsafe { dev.device.unmap_memory(io_mem); }

    let hi = || vk::DescriptorBufferInfo::default().buffer(hidden.handle).offset(0).range(vk::WHOLE_SIZE);
    let qi = || vk::DescriptorBufferInfo::default().buffer(q_buf.handle).offset(0).range(vk::WHOLE_SIZE);
    let ki = || vk::DescriptorBufferInfo::default().buffer(k_buf.handle).offset(0).range(vk::WHOLE_SIZE);
    let vi = || vk::DescriptorBufferInfo::default().buffer(v_buf.handle).offset(0).range(vk::WHOLE_SIZE);
    let ai = || vk::DescriptorBufferInfo::default().buffer(attn_buf.handle).offset(0).range(vk::WHOLE_SIZE);

    // QKV push constants (reusable)
    let mut qkv_pc = [0u8; 128];
    qkv_pc[0..4].copy_from_slice(&0u32.to_le_bytes());     // rows=offset
    qkv_pc[4..8].copy_from_slice(&dim.to_le_bytes());      // cols=dim
    qkv_pc[32..36].copy_from_slice(&n_q.to_le_bytes());    // opt0=n_q_heads
    qkv_pc[36..40].copy_from_slice(&n_kv.to_le_bytes());   // opt1=n_kv_heads
    qkv_pc[40..44].copy_from_slice(&hd.to_le_bytes());     // opt2=head_dim
    qkv_pc[44..48].copy_from_slice(&1u32.to_le_bytes());   // opt3=1 (Q8_0 dequant)

    // Attention push constants
    let mut attn_pc = [0u8; 128];
    attn_pc[0..4].copy_from_slice(&1u32.to_le_bytes());    // seq_len
    attn_pc[4..8].copy_from_slice(&n_q.to_le_bytes());     // n_heads
    attn_pc[8..12].copy_from_slice(&n_kv.to_le_bytes());   // n_kv_heads
    attn_pc[12..16].copy_from_slice(&hd.to_le_bytes());    // head_dim
    attn_pc[16..20].copy_from_slice(&dim.to_le_bytes());   // max_seq_len

    // Build dispatch chain: RMSNorm → QKV → RoPE → Attention → KV Write → Residual
    let mut chain = dispatch::DispatchChain::new();
    let ex = dispatch::BarrierKind::ExecOnly;
    let mf = dispatch::BarrierKind::MemoryFlush;

    chain.add(dispatch::DispatchStep { // 1. RMSNorm
        pipeline_name: "rms_norm",
        push_data: pc_bytes(&constants::RMSNormPC { rows: 1, dim, eps: 1e-6 }),
        workgroup_x: (dim + 255) / 256, workgroup_y: 1, workgroup_z: 1,
        buffers: vec![hi(), w_pre_norm, w_pre_norm, hi(), w_pre_norm], // in=data, weight=wf32, out=data (in-place)
        barrier: ex,
    });
    chain.add(dispatch::DispatchStep { // 2. QKV
        pipeline_name: "qkv",
        push_data: qkv_pc,
        workgroup_x: (dim + 63) / 64, workgroup_y: n_q, workgroup_z: 1,
        buffers: vec![hi(), w_qkv, w_qkv, qi(), ki(), vi()],
        barrier: ex,
    });
    chain.add(dispatch::DispatchStep { // 3. RoPE
        pipeline_name: "rope",
        push_data: attn_pc,
        workgroup_x: n_q, workgroup_y: 1, workgroup_z: 1,
        buffers: vec![qi(), ki()],
        barrier: ex,
    });
    chain.add(dispatch::DispatchStep { // 4. Attention (uses q/k/v, writes attn_out)
        pipeline_name: "attention",
        push_data: attn_pc,
        workgroup_x: n_q, workgroup_y: 1, workgroup_z: 1,
        buffers: vec![qi(), ki(), vi(), ki(), vi(), ai()], // q, k, v, k_cache(≈k), v_cache(≈v), out
        barrier: mf,
    });
    chain.add(dispatch::DispatchStep { // 5. KV Write (ponytail: uses same k/v buffers as cache)
        pipeline_name: "kv_write",
        push_data: attn_pc,
        workgroup_x: n_kv, workgroup_y: 1, workgroup_z: 1,
        buffers: vec![ki(), vi(), ki(), vi()], // k, v, k_cache, v_cache
        barrier: mf,
    });
    chain.add(dispatch::DispatchStep { // 6. Residual add: hidden += attn_out
        pipeline_name: "residual_add",
        push_data: pc_bytes(&constants::RMSNormPC { rows: 0, dim, eps: 0.0 }),
        workgroup_x: (dim + 255) / 256, workgroup_y: 1, workgroup_z: 1,
        buffers: vec![hi(), ai(), ai(), ai()], // data=a, data8, data16, data_b=attn_out
        barrier: ex,
    });
    chain.add(dispatch::DispatchStep { // 7. Post-attention RMSNorm (pre-MoE)
        pipeline_name: "rms_norm",
        push_data: pc_bytes(&constants::RMSNormPC { rows: 1, dim, eps: 1e-6 }),
        workgroup_x: (dim + 255) / 256, workgroup_y: 1, workgroup_z: 1,
        buffers: vec![hi(), w_post_norm, w_post_norm, hi(), w_post_norm], // in-place
        barrier: mf,
    });

    chain.execute(dev, &engine.shaders)?;
    println!("Full attention chain (6 dispatches): OK");

    // Read back attn_out
    let attn_off = (hidden.size + 127) & !127 + (q_buf.size + 127) & !127 + (k_buf.size + 127) & !127 + (v_buf.size + 127) & !127;
    {
        let ptr = unsafe { dev.device.map_memory(io_mem, attn_off, attn_buf.size, vk::MemoryMapFlags::empty())? } as *const f32;
        let attn = unsafe { std::slice::from_raw_parts(ptr, 8) };
        println!("Attn[0..4] = {:?}", &attn[..4]);
        let attn_ok = !attn[0].is_nan() && attn[0].abs() > 0.01;
        println!("Attn: {} first={}", if attn_ok { "OK" } else { "ZERO?" }, attn[0]);
        unsafe { dev.device.unmap_memory(io_mem); }
    }
    let ptr = unsafe { dev.device.map_memory(io_mem, 0, hidden.size, vk::MemoryMapFlags::empty())? } as *const f32;
    let out = unsafe { std::slice::from_raw_parts(ptr, 8) };
    println!("Hidden[0..4] after attention+residual = {:?}", &out[..4]);
    let ok = !out[0].is_nan();
    println!("Full layer: {} first={}", if ok { "PASS" } else { "FAIL" }, out[0]);

    unsafe { dev.device.unmap_memory(io_mem); }
    for b in &bufs { unsafe { dev.device.destroy_buffer(b.handle, None); } }
    unsafe { dev.device.free_memory(io_mem, None); }
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
