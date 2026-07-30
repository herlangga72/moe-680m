use ash::vk;
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
    "ssm_norm", "ssm_conv", "ssm_scan", "ssm_proj", "ssm_out",
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
    // ponytail: one descriptor set layout — all buffers, bound once per forward pass
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

        let set_layouts_ref = [desc_set_layout];
        let push_ranges_ref = [push_range];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts_ref)
            .push_constant_ranges(&push_ranges_ref);

        let pipeline_layout = unsafe {
            dev.device.create_pipeline_layout(&pipeline_layout_info, None)?
        };

        // Descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
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
        for (_i, &name) in SHADERS.iter().enumerate() {
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
