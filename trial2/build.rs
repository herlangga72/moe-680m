use std::process::Command;

fn main() {
    let shader_dir = std::path::Path::new("shaders");
    let out_dir = std::path::Path::new("src/shaders");
    std::fs::create_dir_all(out_dir).unwrap();

    for entry in std::fs::read_dir(shader_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().map_or(false, |e| e == "comp") {
            let name = path.file_stem().unwrap().to_str().unwrap();
            let spv = out_dir.join(format!("{}.spv", name));
            let status = Command::new("glslc")
                .args(&[
                    "--target-env=vulkan1.3",
                    "-fshader-stage=compute",
                    "-I", "shaders",
                    path.to_str().unwrap(),
                    "-o", spv.to_str().unwrap(),
                ])
                .status()
                .expect("glslc not found — install Vulkan SDK");
            assert!(status.success(), "glslc failed for {}", name);
            println!("cargo:rerun-if-changed=shaders/{}.comp", name);
        }
    }
    println!("cargo:rerun-if-changed=shaders/common.glsl");
}
