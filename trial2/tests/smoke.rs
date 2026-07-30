use std::process::Command;

/// Run `--smoke` and assert the binary initialises Vulkan successfully.
#[test]
fn test_smoke_flag() {
    let output = Command::new("./target/release/moe-680m")
        .arg("--smoke")
        .output()
        .expect("failed to run binary --smoke");
    assert!(
        output.status.success(),
        "binary --smoke exited with code {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Vulkan OK"),
        "expected 'Vulkan OK' in stdout, got: {stdout}",
    );
}

/// Running the binary without `--model` should fail.
#[test]
fn test_no_model_error() {
    let output = Command::new("./target/release/moe-680m")
        .output()
        .expect("failed to run binary without args");
    assert!(
        !output.status.success(),
        "expected non-zero exit, got success",
    );
}

/// Run the binary with a real model and generate one token.
///
/// This test requires a GGUF model file at the path below and is excluded
/// from `cargo test` by default.  Run with `cargo test -- --ignored`.
#[test]
#[ignore]
fn test_generate_one_token() {
    let output = Command::new("./target/release/moe-680m")
        .args(["--model", "model/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"])
        .output()
        .expect("failed to run binary with --model");
    assert!(
        output.status.success(),
        "binary with --model exited with code {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
    );
}
