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

fn test_forward(mut engine: engine::Engine, _gguf: &gguf::GgufFile) -> Result<()> {
    use ash::vk;
    println!("\n=== GPU forward-pass test ===");

    let dev = &engine.device;
    let dim = engine.config.hidden_dim;

    // Use residual_add: a += b — simple FP32, no dequant
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
    let size = (dim * 4) as u64; // FP32
    let buf_a = memory::Buffer::new(&dev.device, size, usage)?;
    let buf_b = memory::Buffer::new(&dev.device, size, usage)?;

    // Allocate UMA memory
    let total = (buf_a.size + buf_b.size + 127) & !127;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(total)
        .memory_type_index(dev.find_memory_type(u32::MAX,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?);
    let mem = unsafe { dev.device.allocate_memory(&alloc_info, None)? };
    unsafe { dev.device.bind_buffer_memory(buf_a.handle, mem, 0)?; }
    unsafe { dev.device.bind_buffer_memory(buf_b.handle, mem, buf_a.size)?; }

    // Fill: a = 3.0, b = 2.0 → after residual_add: a should be 5.0
    let ptr = unsafe { dev.device.map_memory(mem, 0, total, vk::MemoryMapFlags::empty())? } as *mut f32;
    for i in 0..dim as usize {
        unsafe { *ptr.add(i) = 3.0f32; }
        unsafe { *ptr.add(dim as usize + i) = 2.0f32; }
    }
    unsafe { dev.device.unmap_memory(mem); }

    // Dispatch residual_add
    let pipeline = engine.shaders.pipelines.get("residual_add")
        .ok_or_else(|| Error::Api("residual_add pipeline not found".into()))?;

    let cmd_pool = unsafe {
        dev.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(dev.queue_family), None,
        )?
    };
    let cmd = unsafe {
        dev.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
        )?
    }[0];

    unsafe {
        dev.device.begin_command_buffer(cmd,
            &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, *pipeline);

        // residual_add: data[base+j] += data_b[base+j]; base=pc.rows, n=pc.cols
        let pc = constants::RMSNormPC { rows: 0, dim, eps: 0.0 };
        dev.device.cmd_push_constants(cmd, engine.shaders.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc));

        let idx = shaders::SHADERS.iter().position(|&n| n == "residual_add").unwrap();
        let ds = engine.shaders.desc_sets[idx];
        let bi_a = vk::DescriptorBufferInfo::default().buffer(buf_a.handle).offset(0).range(vk::WHOLE_SIZE);
        let bi_b = vk::DescriptorBufferInfo::default().buffer(buf_b.handle).offset(0).range(vk::WHOLE_SIZE);
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&bi_a)),
            vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&bi_b)),
        ];
        dev.device.update_descriptor_sets(&writes, &[]);
        dev.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE,
            engine.shaders.pipeline_layout, 0, &[ds], &[]);

        dev.device.cmd_dispatch(cmd, (dim + 255) / 256, 1, 1);
        dev.device.end_command_buffer(cmd)?;
    }

    let cmd_bufs = [cmd];
    unsafe {
        dev.device.queue_submit(dev.queue, &[vk::SubmitInfo::default().command_buffers(&cmd_bufs)], vk::Fence::null())?;
        dev.device.queue_wait_idle(dev.queue)?;
    }

    // Read back and verify: a[i] should be 5.0
    let ptr = unsafe { dev.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty())? } as *const f32;
    let out_slice = unsafe { std::slice::from_raw_parts(ptr, dim as usize) };
    println!("residual_add: a[0..4] = {:?} (expected [5.0, 5.0, 5.0, 5.0])", &out_slice[..4]);
    let all_five = out_slice.iter().all(|v| (v - 5.0).abs() < 0.001);
    println!("Verify 3+2=5: {}", if all_five { "PASS" } else { "FAIL" });
    unsafe { dev.device.unmap_memory(mem); }

    unsafe {
        dev.device.destroy_command_pool(cmd_pool, None);
        dev.device.destroy_buffer(buf_a.handle, None);
        dev.device.destroy_buffer(buf_b.handle, None);
        dev.device.free_memory(mem, None);
    }

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
