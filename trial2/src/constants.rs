use ash::vk;

pub const ALIGNMENT: u64 = 128;

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
pub struct RMSNormPC {
    pub rows: u32,
    pub dim: u32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
pub struct LinearPC {
    pub in_dim: u32,
    pub out_dim: u32,
    pub pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
pub struct AttentionPC {
    pub seq_len: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
pub struct RouterPC {
    pub dim: u32,
    pub n_experts: u32,
    pub n_active: u32,
    pub n_shared: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
pub struct MoEPC {
    pub dim: u32,
    pub intermediate: u32,
    pub expert_idx: u32,
    pub is_shared: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
pub struct SamplePC {
    pub vocab_size: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
pub struct MTPBlockPC {
    pub dim: u32,
    pub head_dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub pos: u32,
    pub block_idx: u32,
}

/// Execution-only barrier (no memory flush): compute -> compute on same queue
pub fn barrier_exec_only() -> vk::MemoryBarrier2<'static> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
    // NO access flags -- execution barrier only
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
