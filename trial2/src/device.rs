use ash::{Entry, Instance, vk};
use crate::error::{Error, Result};

pub struct Device {
    pub instance: Instance,
    pub _entry: Entry,
    pub device: ash::Device,
    pub physical: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub limits: vk::PhysicalDeviceLimits,
    pub subgroup_size: u32,
    pub timestamp_period: f32,
}

impl Device {
    pub fn init() -> Result<Self> {
        let entry = unsafe {
            Entry::load().map_err(|_| Error::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?
        };

        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_3);

        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
        }.map_err(|e| Error::Vulkan(e.into()))?;

        let physical = unsafe { instance.enumerate_physical_devices()? }
            .into_iter()
            .next()
            .ok_or(Error::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;

        let props = unsafe { instance.get_physical_device_properties(physical) };

        // Query subgroup size via VK_KHR_vulkan11 / Vulkan 1.1 properties
        let mut vulkan_11_props = vk::PhysicalDeviceVulkan11Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut vulkan_11_props);
        unsafe {
            instance.get_physical_device_properties2(physical, &mut props2);
        }
        let subgroup_size = vulkan_11_props.subgroup_size; // ponytail: ~64 on RDNA2

        let queue_family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
            .into_iter()
            .enumerate()
            .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .ok_or(Error::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED))?;

        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(queue_family)
                        .queue_priorities(&[1.0])]),
                None,
            )
        }.map_err(|e| Error::Vulkan(e.into()))?;

        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        Ok(Self {
            queue,
            queue_family,
            device,
            physical,
            instance,
            _entry: entry,
            limits: props.limits,
            subgroup_size,
            timestamp_period: props.limits.timestamp_period as f32,
        })
    }

    pub fn find_memory_type(&self, type_filter: u32, flags: vk::MemoryPropertyFlags) -> Result<u32> {
        let mem_props = unsafe {
            self.instance.get_physical_device_memory_properties(self.physical)
        };
        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_props.memory_types[i as usize].property_flags.contains(flags)
            {
                return Ok(i);
            }
        }
        Err(Error::Vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
