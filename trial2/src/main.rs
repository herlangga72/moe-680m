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
                args.max_context = raw[i].parse().unwrap_or(16384);
            }
            "--smoke" => args.smoke = true,
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
    let total_memory = 24u64 * 1024 * 1024 * 1024; // 24 GB
    let arena = memory::Arena::new(
        dev.device.clone(),
        total_memory,
        dev.find_memory_type(u32::MAX, ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)?,
    )?;

    // ---- 5. Create inference engine ----
    let engine = engine::Engine::new(&gguf, arena, dev, shader_cache, args.max_context)?;

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
