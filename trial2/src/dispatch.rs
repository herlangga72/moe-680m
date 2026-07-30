use ash::vk;
use crate::device::Device as GpuDevice;
use crate::error::Result;
use crate::shaders::{ShaderCache, SHADERS};

/// Kind of pipeline barrier to insert after a dispatch step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierKind {
    /// No barrier — next step runs in pipeline order, no guarantees on memory
    None,
    /// Execution barrier only — next step runs after previous completes, but
    /// memory may not be visible yet (same-queue compute-to-compute ordering)
    ExecOnly,
    /// Full memory flush — all previous SSBO writes visible to subsequent reads
    MemoryFlush,
    /// Host-available — makes writes visible to host (for read-back buffers)
    HostRead,
}

/// A single dispatch step in a multi-step chain.
pub struct DispatchStep {
    /// Name matching a key in ShaderCache::pipelines (must be in SHADERS list)
    pub pipeline_name: &'static str,
    /// 128 bytes of push constants (matches the max range in pipeline layout)
    pub push_data: [u8; 128],
    /// Workgroup counts for vkCmdDispatch
    pub workgroup_x: u32,
    pub workgroup_y: u32,
    pub workgroup_z: u32,
    /// Buffer bindings (one per descriptor slot, binding 0, 1, 2, …)
    pub buffers: Vec<vk::DescriptorBufferInfo>,
    /// Barrier to apply after this step
    pub barrier: BarrierKind,
}

/// Accumulates a sequence of compute dispatches and executes them in a single
/// submission with appropriate barriers between steps.
pub struct DispatchChain {
    steps: Vec<DispatchStep>,
}

impl DispatchChain {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Append a dispatch step to the chain.
    pub fn add(&mut self, step: DispatchStep) {
        self.steps.push(step);
    }

    /// Record all accumulated steps into a single command buffer, submit it,
    /// and wait for the queue to become idle.
    ///
    /// Each step:
    ///  1. Binds the compute pipeline from `shaders`
    ///  2. Pushes 128 bytes of constants
    ///  3. Updates the descriptor set for the pipeline's shader with the
    ///     provided buffer bindings and binds it
    ///  4. Dispatches the compute shader
    ///  5. Inserts a pipeline barrier according to `step.barrier`
    pub fn execute(&self, dev: &GpuDevice, shaders: &ShaderCache) -> Result<()> {
        // --- command pool ---
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(dev.queue_family);
        let pool = unsafe { dev.device.create_command_pool(&pool_info, None)? };

        // --- command buffer ---
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffers = unsafe { dev.device.allocate_command_buffers(&alloc_info)? };
        let cmd = cmd_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { dev.device.begin_command_buffer(cmd, &begin_info)? };

        // --- record dispatches ---
        // Pre-construct the three barrier types so they live long enough for
        // DependencyInfo borrows across all loop iterations.
        let exec_barrier = crate::constants::barrier_exec_only();
        let mem_barrier = crate::constants::barrier_memory_flush();
        let host_barrier = crate::constants::barrier_host_read();

        for step in &self.steps {
            // 1. Bind pipeline
            let pipeline = shaders.pipelines.get(step.pipeline_name).ok_or_else(|| {
                crate::error::Error::Api(format!("unknown pipeline: {}", step.pipeline_name))
            })?;
            unsafe {
                dev.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, *pipeline);
            }

            // 2. Push constants
            unsafe {
                dev.device.cmd_push_constants(
                    cmd,
                    shaders.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &step.push_data,
                );
            }

            // 3. Update and bind descriptor set
            let shader_idx = SHADERS.iter().position(|&name| name == step.pipeline_name).ok_or_else(|| {
                crate::error::Error::Api(format!("pipeline '{}' not in SHADERS list", step.pipeline_name))
            })?;
            let desc_set = shaders.desc_sets[shader_idx];

            if !step.buffers.is_empty() {
                let writes: Vec<vk::WriteDescriptorSet> = step
                    .buffers
                    .iter()
                    .enumerate()
                    .map(|(i, buf)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(desc_set)
                            .dst_binding(i as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(buf))
                    })
                    .collect();
                unsafe {
                    dev.device.update_descriptor_sets(&writes, &[]);
                }
            }

            unsafe {
                dev.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    shaders.pipeline_layout,
                    0,
                    &[desc_set],
                    &[],
                );
            }

            // 4. Dispatch
            unsafe {
                dev.device.cmd_dispatch(cmd, step.workgroup_x, step.workgroup_y, step.workgroup_z);
            }

            // 5. Barrier
            if step.barrier != BarrierKind::None {
                let dep_info = match step.barrier {
                    BarrierKind::ExecOnly => {
                        vk::DependencyInfo::default()
                            .memory_barriers(std::slice::from_ref(&exec_barrier))
                    }
                    BarrierKind::MemoryFlush => {
                        vk::DependencyInfo::default()
                            .memory_barriers(std::slice::from_ref(&mem_barrier))
                    }
                    BarrierKind::HostRead => {
                        vk::DependencyInfo::default()
                            .memory_barriers(std::slice::from_ref(&host_barrier))
                    }
                    BarrierKind::None => unreachable!(),
                };
                unsafe {
                    dev.device.cmd_pipeline_barrier2(cmd, &dep_info);
                }
            }
        }

        // --- end recording & submit ---
        unsafe { dev.device.end_command_buffer(cmd)?; }

        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
        unsafe {
            dev.device.queue_submit(dev.queue, &[submit_info], vk::Fence::null())?;
            dev.device.queue_wait_idle(dev.queue)?;
        }

        // --- cleanup ---
        unsafe {
            dev.device.destroy_command_pool(pool, None);
        }

        Ok(())
    }
}
