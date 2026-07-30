#[test]
fn parse_model_metadata() {
    let path = std::path::Path::new("models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf");
    if !path.exists() {
        eprintln!("SKIP: model not found");
        return;
    }
    let gguf = moe_680m::gguf::GgufFile::open(path).expect("GGUF open");
    
    eprintln!("=== METADATA KEYS ===");
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for k in keys.iter().take(80) {
        eprintln!("  {} = {:?}", k, gguf.metadata.get(*k).unwrap());
    }
    
    eprintln!("\n=== ALL TENSOR NAMES ===");
    for t in &gguf.tensors {
        eprintln!("  {}", t.name);
    }
    
    eprintln!("\n=== MODEL CONFIG ===");
    match gguf.model_config() {
        Ok(c) => {
            eprintln!("  n_layers={} dim={} heads_q={} heads_kv={} head_dim={}",
                c.n_layers, c.hidden_dim, c.n_heads_q, c.n_heads_kv, c.head_dim);
            eprintln!("  ffn={} experts={}/{}+{} vocab={} ctx={}",
                c.ffn_intermediate, c.n_experts, c.n_active_experts, c.n_shared_experts,
                c.vocab_size, c.max_seq_len);
            eprintln!("  rope_theta={} rope_type={} mtp={}/{} eps={}",
                c.rope_theta, c.rope_type, c.n_mtp_modules, c.mtp_depth, c.eps);
        },
        Err(e) => eprintln!("  CONFIG ERROR: {}", e),
    }
    
    // At minimum, verify model loads
    assert!(gguf.tensors.len() > 100, "expected >100 tensors, got {}", gguf.tensors.len());
    assert!(gguf.metadata.len() > 10, "expected >10 metadata keys, got {}", gguf.metadata.len());
}
