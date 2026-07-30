// Weight buffer pool — layer-by-layer streaming upload from GGUF.
// On UMA (iGPU), weights are uploaded to HOST_VISIBLE buffers from the mmap'd GGUF.
// Buffers are reused across layers to keep GPU memory footprint low (~100 MB).
//
// ponytail: single VkDeviceMemory pool, pre-allocated for max layer weight size.

use ash::vk;
use std::collections::HashMap;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::gguf::{GgufDtype, GgufFile, TensorInfo};
use crate::memory::Buffer;

pub struct WeightPool {
    device: ash::Device,
    memory: vk::DeviceMemory,
    buffers: HashMap<String, (vk::Buffer, u64, u64)>, // name -> (handle, offset, size)
    pub mapped_ptr: *mut u8,
    total_size: u64,
    next_offset: u64,
}

impl WeightPool {
    /// Allocate a pool large enough for one layer's weights (~100 MB).
    /// Ponytail: 128 MB, RDNA2 UMA, layer weight set fits.
    pub fn new(dev: &Device, size: u64) -> Result<Self> {
        let mem_type = dev.find_memory_type(
            u32::MAX,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(mem_type);
        let memory = unsafe { dev.device.allocate_memory(&alloc_info, None)? };

        let ptr = unsafe {
            dev.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?
        } as *mut u8;

        Ok(Self {
            device: dev.device.clone(),
            memory,
            buffers: HashMap::new(),
            mapped_ptr: ptr,
            total_size: size,
            next_offset: 0,
        })
    }

    /// Upload a tensor from GGUF into the pool, return a DescriptorBufferInfo.
    /// Reuses existing buffer if already uploaded for this layer.
    /// Upload a tensor and return a DescriptorBufferInfo pointing to it.
    /// The buffer is bound at the pool offset; DescriptorBufferInfo offset is 0.
    pub fn upload(&mut self, gguf: &GgufFile, tensor_name: &str) -> Result<vk::DescriptorBufferInfo> {
        if let Some(&(buf, _offset, size)) = self.buffers.get(tensor_name) {
            return Ok(vk::DescriptorBufferInfo::default()
                .buffer(buf).offset(0).range(size));
        }

        let tensor = gguf.find_tensor(tensor_name)
            .ok_or_else(|| Error::Api(format!("tensor not found: {}", tensor_name)))?;
        let raw = gguf.tensor_data(tensor);
        let size = raw.len() as u64;
        let copy_offset = if tensor.dtype == GgufDtype::F32 && raw.len() > 16 {
            // ponytail: Unsloth IQ4_XS GGUF stores 16-byte garbage prefix on F32 norm tensors.
            16usize
        } else { 0usize };
        let copy_len = raw.len() - copy_offset;
        if copy_len == 0 { return Err(Error::Api(format!("zero-sized tensor: {}", tensor_name))); }

        // Align to 128 bytes
        let offset = (self.next_offset + 127) & !127;
        if offset + size > self.total_size {
            return Err(Error::OutOfMemory { needed: size, available: self.total_size - offset });
        }

        // Create buffer
        let buf_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buf = unsafe { self.device.create_buffer(&buf_info, None)? };
        unsafe { self.device.bind_buffer_memory(buf, self.memory, offset)?; }

        // Copy data (skip prefix if needed)
        unsafe {
            std::ptr::copy_nonoverlapping(
                raw.as_ptr().add(copy_offset),
                self.mapped_ptr.add(offset as usize),
                copy_len,
            );
        }

        self.buffers.insert(tensor_name.to_string(), (buf, offset, size));
        self.next_offset = offset + size;

        Ok(vk::DescriptorBufferInfo::default().buffer(buf).offset(0).range(size))
    }

    /// Clear all buffers for next layer (reuse memory from start).
    pub fn clear(&mut self) {
        for &(buf, _, _) in self.buffers.values() {
            unsafe { self.device.destroy_buffer(buf, None); }
        }
        self.buffers.clear();
        self.next_offset = 0;
    }

    pub fn destroy(&mut self) {
        self.clear();
        unsafe {
            self.device.unmap_memory(self.memory);
            self.device.free_memory(self.memory, None);
        }
    }
}
