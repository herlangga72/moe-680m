use ash::{Device, vk};
use std::collections::HashMap;
use crate::constants::ALIGNMENT;
use crate::error::{Error, Result};

fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

pub struct Buffer {
    pub handle: vk::Buffer,
    pub size: u64,
}

impl Buffer {
    pub fn new(device: &Device, size: u64, usage: vk::BufferUsageFlags) -> Result<Self> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { device.create_buffer(&create_info, None)? };
        Ok(Self { handle, size })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe { device.destroy_buffer(self.handle, None); }
    }
}

pub struct Arena {
    device: Device,
    memory: vk::DeviceMemory,
    total_size: u64,
    offsets: HashMap<String, (u64, u64)>,
    next_offset: u64,
}

impl Arena {
    pub fn new(
        device: Device,
        size: u64,
        memory_type_index: u32,
    ) -> Result<Self> {
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };

        Ok(Self {
            device,
            memory,
            total_size: size,
            offsets: HashMap::new(),
            next_offset: 0,
        })
    }

    pub fn allocate(&mut self, name: &str, size: u64) -> Result<u64> {
        let offset = align_up(self.next_offset, ALIGNMENT);
        if offset + size > self.total_size {
            return Err(Error::OutOfMemory {
                needed: size,
                available: self.total_size - offset,
            });
        }
        self.offsets.insert(name.to_string(), (offset, size));
        self.next_offset = offset + size;
        Ok(offset)
    }

    pub fn bind_buffer(&self, name: &str, buffer: &Buffer) -> Result<()> {
        let &(offset, _) = self.offsets.get(name)
            .ok_or_else(|| Error::Api(format!("arena: no allocation for '{}'", name)))?;
        unsafe {
            self.device.bind_buffer_memory(buffer.handle, self.memory, offset)?;
        }
        Ok(())
    }

    pub fn destroy(&mut self) {
        unsafe { self.device.free_memory(self.memory, None); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 128), 0);
        assert_eq!(align_up(1, 128), 128);
        assert_eq!(align_up(128, 128), 128);
        assert_eq!(align_up(129, 128), 256);
    }
}
