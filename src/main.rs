mod device;
mod memory;
mod gguf;
mod pipeline;
mod inference;
mod router;
mod sampling;
mod tokenizer;

use std::env;

pub struct DebugContext {
    pub enabled: bool,
    pub vk_validation: bool,
}

impl DebugContext {
    fn from_env() -> Self {
        let de = env::var("MOE_DEBUG").ok();
        let ve = env::var("MOE_VK_VALIDATION").ok();
        Self {
            enabled: de.map_or(false, |v| v != "0"),
            vk_validation: ve.map_or(false, |v| v != "0"),
        }
    }
}

fn print_help() {
    eprintln!("moe-680m — MoE inference engine for Radeon 680M");
    eprintln!();
    eprintln!("Usage: moe-680m [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --debug              Debug output (or MOE_DEBUG=1)");
    eprintln!("  --validate           Vulkan validation layers (MOE_VK_VALIDATION=1)");
    eprintln!("  --smoke              Run Vulkan smoke test");
    eprintln!("  --model <PATH>       Load GGUF model");
    eprintln!("  --prompt <TEXT>      Run inference prompt");
    eprintln!("  --max-tokens <N>     Max tokens to generate");
    eprintln!("  --server [PORT]      Start HTTP server (default port 8080)");
    eprintln!("  --help               Print this help");
}

fn main() {
    let mut debug = DebugContext::from_env();
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    let mut flag_smoke = false;
    let mut model_path = None::<String>;
    let mut prompt = None::<String>;
    let mut max_tokens = 100u32;
    let mut server_port = None::<u16>;

    while i < args.len() {
        match args[i].as_str() {
            "--debug" | "-d" => debug.enabled = true,
            "--validate" => debug.vk_validation = true,
            "--smoke" => flag_smoke = true,
            "--model" => { i += 1; model_path = Some(args.get(i).cloned().unwrap_or_default()); }
            "--prompt" => { i += 1; prompt = Some(args.get(i).cloned().unwrap_or_default()); }
            "--max-tokens" => { i += 1; max_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(100); }
            "--server" => {
                let port = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(8080u16);
                server_port = Some(port);
            }
            "--help" | "-h" => { print_help(); return; }
            _ => { eprintln!("Unknown: {}", args[i]); print_help(); return; }
        }
        i += 1;
    }

    if flag_smoke || (model_path.is_none() && server_port.is_none() && prompt.is_none()) {
        run_smoke(&debug);
        return;
    }

    if let Some(path) = model_path {
        run_inference(&path, prompt.as_deref(), max_tokens, &debug, server_port);
    }
}

fn run_smoke(debug: &DebugContext) {
    eprintln!("moe-680m smoke test");
    match device::DeviceContext::init(debug) {
        Ok(dc) => {
            eprintln!("✅ Device: {} | Driver: {} | UMA type: {} | Queue: {}",
                dc.device_name, dc.driver_version, dc.uma_memory_type, dc.queue_family);
            eprintln!("   maxComputeSharedMemorySize: {} KB", dc.max_compute_shared_memory_size / 1024);
            unsafe {
                match dc.allocate_uma(1024 * 1024) {
                    Ok((mem, ptr)) => {
                        std::ptr::write_bytes(ptr, 0xAB, 1024 * 1024);
                        let v = std::ptr::read(ptr);
                        eprintln!("✅ UMA write/read: {} (expected 0xAB)", if v == 0xAB { "OK" } else { "FAIL" });
                        dc.device.free_memory(mem, None);
                    }
                    Err(e) => eprintln!("❌ UMA alloc: {}", e),
                }
            }
        }
        Err(e) => eprintln!("❌ Vulkan: {}", e),
    }
}

// ── Helpers to extract tokenizer data from GGUF metadata ──

fn meta_get_str(meta: &std::collections::HashMap<String, gguf::MetadataValue>, key: &str) -> Option<String> {
    meta.get(key).and_then(|v| {
        if let gguf::MetadataValue::String(s) = v { Some(s.clone()) } else { None }
    })
}

fn meta_get_arr_str(meta: &std::collections::HashMap<String, gguf::MetadataValue>, key: &str) -> Option<Vec<String>> {
    meta.get(key).and_then(|v| {
        if let gguf::MetadataValue::Array(arr) = v {
            let strs: Vec<String> = arr.iter().filter_map(|x| {
                if let gguf::MetadataValue::String(s) = x { Some(s.clone()) } else { None }
            }).collect();
            if strs.len() == arr.len() { Some(strs) } else { None }
        } else { None }
    })
}

fn meta_get_arr_f32(meta: &std::collections::HashMap<String, gguf::MetadataValue>, key: &str) -> Option<Vec<f32>> {
    meta.get(key).and_then(|v| {
        if let gguf::MetadataValue::Array(arr) = v {
            let vals: Vec<f32> = arr.iter().filter_map(|x| {
                match x {
                    gguf::MetadataValue::Float32(f) => Some(*f),
                    gguf::MetadataValue::Float64(f) => Some(*f as f32),
                    _ => None,
                }
            }).collect();
            if vals.len() == arr.len() { Some(vals) } else { None }
        } else { None }
    })
}

fn meta_get_int(meta: &std::collections::HashMap<String, gguf::MetadataValue>, key: &str) -> Option<u32> {
    meta.get(key).map(|v| match v {
        gguf::MetadataValue::Uint32(x) => *x,
        gguf::MetadataValue::Int32(x) => *x as u32,
        gguf::MetadataValue::Uint64(x) => *x as u32,
        _ => 0,
    })
}

// ── Inference ──

fn run_inference(model_path: &str, prompt_text: Option<&str>, max_tokens: u32,
                 debug: &DebugContext, server_port: Option<u16>) {
    // 1. Memory-map GGUF
    eprintln!("Loading model: {}", model_path);
    let file = match std::fs::File::open(model_path) {
        Ok(f) => f,
        Err(e) => { eprintln!("❌ Open file: {}", e); return; }
    };
    let mmap = match unsafe { memmap2::Mmap::map(&file) } {
        Ok(m) => m,
        Err(e) => { eprintln!("❌ Mmap: {}", e); return; }
    };
    drop(file);

    // 2. Parse GGUF
    let reader = match gguf::GgufReader::parse(&mmap) {
        Ok(r) => r,
        Err(e) => { eprintln!("❌ GGUF parse: {}", e); return; }
    };
    eprintln!("✅ Model: {}", reader.config);

    // Extract config, tensors, metadata before dropping reader
    let config = reader.config.clone();
    let tensors = reader.tensors.clone();
    let tensor_count = reader.tensor_count;
    let tensor_data_offset = reader.tensor_data_offset;
    let meta = &reader.metadata;

    // (NEW) Initialize tokenizer from GGUF metadata
    eprintln!("Initializing tokenizer...");
    let tokenizer_data = match tokenizer::TokenizerData::from_gguf_meta(
        &|k| meta_get_str(meta, k),
        &|k| meta_get_arr_str(meta, k),
        &|k| meta_get_arr_f32(meta, k),
        &|k| meta_get_int(meta, k),
    ) {
        Ok(td) => td,
        Err(e) => { eprintln!("❌ Tokenizer data: {}", e); return; }
    };
    let tokenizer = match tokenizer::Tokenizer::from_data(&tokenizer_data) {
        Ok(t) => t,
        Err(e) => { eprintln!("❌ Tokenizer init: {}", e); return; }
    };
    eprintln!("✅ Tokenizer vocab: {}", tokenizer.vocab_size());

    // 3. Init Vulkan
    eprintln!("Initializing Vulkan...");
    let dc = match device::DeviceContext::init(debug) {
        Ok(d) => d,
        Err(e) => { eprintln!("❌ Vulkan: {}", e); return; }
    };

    // 4. Arena layout
    let weights_size = tensors.iter().map(|t| t.size).sum::<u64>();
    let layout = memory::ArenaLayout::compute(&config, weights_size);

    // 5. Allocate arena
    eprintln!("Allocating arena: {} MB total", layout.total_size / (1024 * 1024));
    let (arena_mem, arena_ptr) = unsafe {
        match dc.allocate_uma(layout.total_size) {
            Ok(x) => x,
            Err(e) => { eprintln!("❌ Arena: {}", e); return; }
        }
    };

    // 6. Load weights
    let reg = memory::TensorRegistry::from_tensors(&tensors, layout.weights_base);
    unsafe {
        for ti in &tensors {
            if let Some(entry) = reg.lookup(&ti.name) {
                let src_start = (tensor_data_offset + ti.offset) as usize;
                let src = &mmap[src_start..src_start + ti.size as usize];
                let dst = arena_ptr.add(entry.arena_offset as usize);
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, ti.size as usize);
                libc::madvise(mmap.as_ptr().add(src_start) as *mut libc::c_void,
                    ti.size as libc::size_t, libc::MADV_DONTNEED);
            }
        }
    }
    let weights = memory::LayerWeights::from_registry(&reg, &config);
    eprintln!("✅ Weights: {} layers, {} experts", weights.num_layers, weights.num_experts);
    drop(mmap);

    // Zero DeltaNet state
    if layout.deltanet_state_size > 0 {
        unsafe { std::ptr::write_bytes(arena_ptr.add(layout.deltanet_state_base as usize),
            0, layout.deltanet_state_size as usize); }
    }

    // 7. Create pipelines
    eprintln!("Creating pipelines...");
    let pipelines = unsafe {
        match pipeline::create_pipelines(&dc.device, debug) {
            Ok(p) => p,
            Err(e) => { eprintln!("❌ Pipelines: {}", e); return; }
        }
    };

    // 8. Bind arena descriptor
    let arena_buffer = unsafe {
        match dc.create_buffer_from_memory(arena_mem, layout.total_size) {
            Ok(b) => b,
            Err(e) => { eprintln!("❌ Arena buffer: {}", e); return; }
        }
    };
    unsafe { pipeline::bind_arena_descriptor(&dc.device, pipelines.desc_set, arena_buffer, layout.total_size); }

    // 9. Init engine
    eprintln!("Initializing engine...");
    let mut engine = match inference::InferenceEngine::new(dc, pipelines, layout, weights, arena_ptr) {
        Ok(e) => e,
        Err(err) => { eprintln!("❌ Engine: {}", err); return; }
    };

    // 10. Run inference
    if let Some(prompt_text) = prompt_text {
        eprintln!("Prompt: {}", prompt_text);
        let input_ids = tokenizer.encode(prompt_text);
        eprintln!("Tokenized: {} tokens", input_ids.len());

        let mut state = inference::InferenceState::new();
        let mut output_ids = Vec::new();
        let mut decoded = String::new();

        // Prefill + generation loop
        for t in 0..max_tokens {
            match engine.generate(&input_ids, &mut state) {
                Ok(token) => {
                    output_ids.push(token);

                    // Decode incremental text
                    let text = tokenizer.decode(&[token]);
                    decoded.push_str(&text);
                    eprint!("{}", text);

                    // Stop at EOS
                    if token == tokenizer.eos_id { break; }
                }
                Err(e) => { eprintln!("\n❌ Inference: {}", e); break; }
            }
        }
        eprintln!("\n✅ Generated {} tokens", output_ids.len());

        unsafe { engine.device.device.free_memory(arena_mem, None); }
    }
}
