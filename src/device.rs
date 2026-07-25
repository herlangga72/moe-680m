use ash::vk;
use std::ffi::CString;

/// Vulkan device context with UMA detection.
pub struct DeviceContext {
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub uma_memory_type: u32,
    pub max_compute_shared_memory_size: u32,
    pub device_name: String,
    pub driver_version: String,
    pub has_buffer_device_address: bool,
}

impl DeviceContext {
    /// Initialize Vulkan, find UMA device, create logical device.
    #[allow(unused_assignments)]
    pub fn init(debug: &crate::DebugContext) -> Result<Self, String> {
        let entry =
            unsafe { ash::Entry::load().map_err(|e| format!("Failed to load Vulkan: {}", e))? };

        // ── Instance ──
        let app_name = CString::new("moe-680m").unwrap();
        let engine_name = CString::new("moe-680m").unwrap();
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 0, 1, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 0, 1, 0),
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|e| format!("Failed to create Vulkan instance: {}", e))?
        };

        // ── Physical devices ──
        let phys_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(|e| format!("enumerate_physical_devices: {}", e))?
        };

        if phys_devices.is_empty() {
            return Err("No Vulkan physical devices found".into());
        }

        let mut selected = None;

        for &pd in &phys_devices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };
            let raw_name: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    props.device_name.as_ptr() as *const u8,
                    props.device_name.len(),
                )
            };
            let name = String::from_utf8_lossy(raw_name)
                .trim_end_matches('\0')
                .to_string();

            let uma = find_uma_memory_type(&mem_props);
            let qf = find_compute_queue_family(&instance, pd);

            if debug.enabled {
                eprintln!(
                    "[moe] Device: {} | UMA={} queue={} maxLDS={}KB",
                    name,
                    uma.map(|i| i.to_string())
                        .unwrap_or_else(|| "none".into()),
                    qf.map(|i| i.to_string())
                        .unwrap_or_else(|| "none".into()),
                    props.limits.max_compute_shared_memory_size / 1024,
                );
                for i in 0..mem_props.memory_type_count {
                    let mt = mem_props.memory_types[i as usize];
                    let heap_size = mem_props.memory_heaps[mt.heap_index as usize].size;
                    eprintln!(
                        "[moe]   mem type {}: flags={:?} heap={} ({}MB)",
                        i,
                        mt.property_flags,
                        mt.heap_index,
                        heap_size / (1024 * 1024)
                    );
                }
            }

            if let (Some(uma_idx), Some(qf_idx)) = (uma, qf) {
                if debug.enabled {
                    eprintln!("[moe]   ✓ Selected (UMA type {})", uma_idx);
                }
                selected = Some((
                    pd,
                    name,
                    vk::PhysicalDeviceProperties { ..props },
                    uma_idx,
                    qf_idx,
                ));
                break;
            }
        }

        let (physical_device, device_name, props, uma_memory_type, queue_family) =
            selected.ok_or_else(|| {
                "No device with UMA memory (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT) found."
                    .to_string()
            })?;

        // ── Logical device ──
        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        // Check for VK_KHR_buffer_device_address
        let ext_props = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap_or_default()
        };

        let has_bda = ext_props.iter().any(|e| {
            let s = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }
                .to_str()
                .unwrap_or("");
            s == "VK_KHR_buffer_device_address"
        });

        let mut extension_name_ptrs: Vec<*const i8> = Vec::new();
        if has_bda {
            let bda_name = CString::new("VK_KHR_buffer_device_address").unwrap();
            // Leak is fine — lives for device lifetime
            extension_name_ptrs.push(bda_name.into_raw() as *const i8);
        }

        // Build device create info with p_next chain
        let mut features16 =
            vk::PhysicalDevice16BitStorageFeatures {
                storage_buffer16_bit_access: vk::TRUE,
                ..Default::default()
            };
        let mut features_f16 =
            vk::PhysicalDeviceShaderFloat16Int8Features {
                shader_float16: vk::TRUE,
                ..Default::default()
            };
        let mut features_bda =
            vk::PhysicalDeviceBufferDeviceAddressFeatures {
                buffer_device_address: if has_bda { vk::TRUE } else { vk::FALSE },
                ..Default::default()
            };

        // Chain features via p_next
        let features16_ptr: *mut std::ffi::c_void = &mut features16 as *mut _ as *mut _;
        let features_f16_ptr: *mut std::ffi::c_void = &mut features_f16 as *mut _ as *mut _;
        let features_bda_ptr: *mut std::ffi::c_void = &mut features_bda as *mut _ as *mut _;

        // Build: create_info → features16 → features_f16 → features_bda
        if has_bda {
            features_f16.p_next = features_bda_ptr;
            features16.p_next = features_f16_ptr;
        } else {
            features16.p_next = features_f16_ptr;
        }

        let mut create_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: extension_name_ptrs.len() as u32,
            pp_enabled_extension_names: extension_name_ptrs.as_ptr(),
            p_next: features16_ptr,
            ..Default::default()
        };

        let device = unsafe {
            instance
                .create_device(physical_device, &create_info, None)
                .map_err(|e| format!("Failed to create logical device: {}", e))?
        };

        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        Ok(DeviceContext {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family,
            uma_memory_type,
            max_compute_shared_memory_size: props.limits.max_compute_shared_memory_size,
            device_name,
            driver_version: format!(
                "{}.{}.{}",
                vk::api_version_variant(props.driver_version),
                vk::api_version_major(props.driver_version),
                vk::api_version_minor(props.driver_version)
            ),
            has_buffer_device_address: has_bda,
        })
    }

    /// Allocate a chunk of UMA memory and map it.
    pub unsafe fn allocate_uma(
        &self,
        size: u64,
    ) -> Result<(vk::DeviceMemory, *mut u8), String> {
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: size,
            memory_type_index: self.uma_memory_type,
            ..Default::default()
        };

        let memory = self
            .device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| {
                format!("Failed to allocate UMA ({} MB): {}", size / (1024 * 1024), e)
            })?;

        let ptr = self
            .device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("Failed to map UMA: {}", e))? as *mut u8;

        Ok((memory, ptr))
    }

    /// Create a storage buffer spanning an existing memory allocation.
    pub unsafe fn create_buffer_from_memory(
        &self,
        memory: vk::DeviceMemory,
        size: u64,
    ) -> Result<vk::Buffer, String> {
        let info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = self
            .device
            .create_buffer(&info, None)
            .map_err(|e| format!("Create buffer: {}", e))?;
        self.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| format!("Bind buffer memory: {}", e))?;
        Ok(buffer)
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn find_uma_memory_type(mem_props: &vk::PhysicalDeviceMemoryProperties) -> Option<u32> {
    let required = vk::MemoryPropertyFlags::DEVICE_LOCAL
        | vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT;

    for i in 0..mem_props.memory_type_count {
        if mem_props.memory_types[i as usize]
            .property_flags
            .contains(required)
        {
            return Some(i);
        }
    }
    None
}

fn find_compute_queue_family(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    for (i, f) in families.iter().enumerate() {
        if f.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            return Some(i as u32);
        }
    }
    None
}
