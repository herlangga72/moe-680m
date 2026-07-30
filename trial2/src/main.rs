mod constants;
mod device;
mod error;
mod gguf;
mod memory;
mod shaders;

use error::Result;

fn smoke() -> Result<()> {
    let mut dev = device::Device::init()?;
    let name = unsafe {
        dev.instance.get_physical_device_properties(dev.physical)
    };
    let name_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            name.device_name.as_ptr() as *const u8,
            name.device_name.len(),
        )
    };
    let name_str = std::str::from_utf8(name_bytes)
        .unwrap_or("unknown")
        .trim_end_matches('\0');
    println!("GPU: {} (subgroup={}, timestamp_period={:.0}ns)",
        name_str,
        dev.subgroup_size,
        dev.timestamp_period,
    );
    println!("Max shared memory: {} KB", dev.limits.max_compute_shared_memory_size / 1024);
    println!("Max workgroup: {}x{}x{}",
        dev.limits.max_compute_work_group_size[0],
        dev.limits.max_compute_work_group_size[1],
        dev.limits.max_compute_work_group_size[2],
    );
    dev.destroy();
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--smoke" {
        return smoke();
    }
    println!("moe-680m v0.2.0 — Qwen 3.6 35B A3B MTP");
    Ok(())
}
