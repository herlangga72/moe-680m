// Job 5: Compute pipeline creation
// Loads SPIR-V, creates shader modules, creates compute pipelines.
// All pipelines share one layout + descriptor set.

use ash::vk;
use std::ffi::CStr;
use std::ptr;

macro_rules! debug_log {
    ($ctx:expr, $($arg:tt)+) => {
        if $ctx.enabled { eprintln!("[moe] {}", format_args!($($arg)+)); }
    };
}

pub const MAX_PIPELINES: usize = 20;

/// Pipeline type enum — direct index into `pipelines[]`.
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum PipelineType {
    RmsNorm = 0,
    DeltaNetQkv = 1,
    DeltaNetStep = 2,
    DeltaNetOutput = 3,
    GqaQkv = 4,
    GqaAttention = 5,
    AttnOutput = 6,
    W1W3Fused = 7,
    W2 = 8,
    Router = 9,
    RouterTopk = 10,
    MoeCombine = 11,
    W2Scatter = 12,
    KvWrite = 13,
    ResidualAdd = 14,
    SiluMult = 15,
    Rope = 16,
}

pub struct PipelineResources {
    pub pipelines: [vk::Pipeline; MAX_PIPELINES],
    pub pipeline_layout: vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub desc_pool: vk::DescriptorPool,
    pub desc_set: vk::DescriptorSet,
    pub pipeline_cache: vk::PipelineCache,
}

/// Create all graphics pipeline infrastructure.
pub unsafe fn create_pipelines(
    device: &ash::Device,
    debug: &crate::DebugContext,
) -> Result<PipelineResources, String> {
    // ── Descriptor set layout (1 storage buffer) ──
    let bindings = [vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        ..Default::default()
    }];
    let desc_layout_info = vk::DescriptorSetLayoutCreateInfo {
        binding_count: 1,
        p_bindings: bindings.as_ptr(),
        ..Default::default()
    };
    let desc_set_layout = device
        .create_descriptor_set_layout(&desc_layout_info, None)
        .map_err(|e| format!("Failed to create descriptor set layout: {}", e))?;

    // ── Descriptor pool + set ──
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo {
        pool_size_count: 1,
        p_pool_sizes: pool_sizes.as_ptr(),
        max_sets: 1,
        ..Default::default()
    };
    let desc_pool = device
        .create_descriptor_pool(&pool_info, None)
        .map_err(|e| format!("Failed to create descriptor pool: {}", e))?;

    let alloc_info = vk::DescriptorSetAllocateInfo {
        descriptor_pool: desc_pool,
        descriptor_set_count: 1,
        p_set_layouts: &desc_set_layout,
        ..Default::default()
    };
    let desc_set = device
        .allocate_descriptor_sets(&alloc_info)
        .map_err(|e| format!("Failed to allocate descriptor set: {}", e))?
        [0];

    // ── Pipeline layout (shared, 128-byte push constant range) ──
    let push_ranges = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: 128,
    }];
    let layout_info = vk::PipelineLayoutCreateInfo {
        set_layout_count: 1,
        p_set_layouts: &desc_set_layout,
        push_constant_range_count: 1,
        p_push_constant_ranges: push_ranges.as_ptr(),
        ..Default::default()
    };
    let pipeline_layout = device
        .create_pipeline_layout(&layout_info, None)
        .map_err(|e| format!("Failed to create pipeline layout: {}", e))?;

    // ── Pipeline cache (load from temp dir if available) ──
    let cache_path = std::env::temp_dir().join("moe_pipeline_cache.bin");
    let (cache_data, _cache_loaded) = (std::fs::read(&cache_path).unwrap_or_default(), false);
    let cache_info = vk::PipelineCacheCreateInfo {
        initial_data_size: cache_data.len(),
        p_initial_data: if cache_data.is_empty() { ptr::null() } else { cache_data.as_ptr() as *const _ },
        ..Default::default()
    };
    let pipeline_cache = device
        .create_pipeline_cache(&cache_info, None)
        .map_err(|e| format!("Failed to create pipeline cache: {}", e))?;
    if !cache_data.is_empty() {
        debug_log!(debug, "  Pipeline cache loaded: {} bytes", cache_data.len());
    }

    // ── Helper: load SPV, create module, create pipeline ──
    let entry = CStr::from_bytes_with_nul(b"main\0").unwrap();

    let create_one = |spv_bytes: &[u8], layout: vk::PipelineLayout|
        -> Result<vk::Pipeline, String> {
        // SPIR-V binary: bytes to u32 slice
        let word_count = spv_bytes.len() / 4;
        let spv_words: &[u32] = std::slice::from_raw_parts(
            spv_bytes.as_ptr() as *const u32, word_count);

        let shader_info = vk::ShaderModuleCreateInfo {
            code_size: spv_words.len() * 4,
            p_code: spv_words.as_ptr(),
            ..Default::default()
        };
        let shader_module = device
            .create_shader_module(&shader_info, None)
            .map_err(|e| format!("Shader module: {}", e))?;

        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: entry.as_ptr(),
            ..Default::default()
        };
        let create_info = vk::ComputePipelineCreateInfo {
            stage,
            layout,
            ..Default::default()
        };
        let result = device.create_compute_pipelines(
            pipeline_cache, &[create_info], None);
        device.destroy_shader_module(shader_module, None);

        match result {
            Ok(p) => Ok(p[0]),
            Err((_, e)) => Err(format!("{:?}", e)),
        }
    };

    // ── Create each available pipeline ──
    let mut pipelines = [vk::Pipeline::null(); MAX_PIPELINES];

    macro_rules! try_pipeline {
        ($pt:expr, $name:expr) => {{
            let idx = $pt as usize;
            match include_bytes!(concat!("shaders/", $name, ".spv")) {
                spv if !spv.is_empty() => {
                    match create_one(spv, pipeline_layout) {
                        Ok(p) => {
                            pipelines[idx] = p;
                            debug_log!(debug, "  Pipeline {}: OK", $name);
                        }
                        Err(e) => {
                            debug_log!(debug, "  Pipeline {}: FAILED: {}", $name, e);
                        }
                    }
                }
                _ => {
                    debug_log!(debug, "  Pipeline {}: SKIPPED (no SPV)", $name);
                }
            }
        }};
    }

    try_pipeline!(PipelineType::RmsNorm, "rms_norm");
    try_pipeline!(PipelineType::DeltaNetQkv, "deltanet_qkv");
    try_pipeline!(PipelineType::DeltaNetStep, "deltanet_step");
    try_pipeline!(PipelineType::DeltaNetOutput, "deltanet_output");
    try_pipeline!(PipelineType::GqaQkv, "qkv");
    try_pipeline!(PipelineType::GqaAttention, "attention");
    try_pipeline!(PipelineType::AttnOutput, "attn_output");
    try_pipeline!(PipelineType::W1W3Fused, "w1_w3_fused");
    try_pipeline!(PipelineType::W2, "w2");
    try_pipeline!(PipelineType::W2Scatter, "w2_scatter");
    try_pipeline!(PipelineType::Router, "router");
    try_pipeline!(PipelineType::RouterTopk, "router_topk");
    try_pipeline!(PipelineType::MoeCombine, "moe_combine");
    try_pipeline!(PipelineType::KvWrite, "kv_write");
    try_pipeline!(PipelineType::ResidualAdd, "residual_add");
    try_pipeline!(PipelineType::SiluMult, "silu_mult");
    try_pipeline!(PipelineType::Rope, "rope");

    // Save pipeline cache
    if let Ok(data) = device.get_pipeline_cache_data(pipeline_cache) {
        if !data.is_empty() {
            let _ = std::fs::write(&cache_path, &data);
        }
    }

    Ok(PipelineResources {
        pipelines,
        pipeline_layout,
        desc_set_layout,
        desc_pool,
        desc_set,
        pipeline_cache,
    })
}

/// Bind the arena buffer to descriptor set slot 0.
/// Must be called once after arena allocation.
pub unsafe fn bind_arena_descriptor(
    device: &ash::Device,
    desc_set: vk::DescriptorSet,
    buffer: vk::Buffer,
    size: u64,
) {
    let buf_info = vk::DescriptorBufferInfo {
        buffer,
        offset: 0,
        range: size,
    };
    let write = vk::WriteDescriptorSet {
        dst_set: desc_set,
        dst_binding: 0,
        descriptor_count: 1,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        p_buffer_info: &buf_info,
        ..Default::default()
    };
    device.update_descriptor_sets(&[write], &[]);
}
