// Job 3: Arena allocator (memory layout + Vulkan allocation)
// Job 4: Weight loader (copy GGUF tensors into arena)
// See plans/data-structures.md for struct layouts.

use crate::gguf::{GgufReader, ModelConfig, TensorInfo};
use crate::util::{f16_bits_to_f32, f32_to_f16_bits};

// ── Arena Layout (pure math, no Vulkan) ──

/// All arena region offsets, computed from ModelConfig.
/// All values in bytes. All aligned to 256 bytes.
pub struct ArenaLayout {
    pub weights_base: u64,
    pub hidden_ping: u64,
    pub hidden_pong: u64,
    pub kv_cache_base: u64,
    pub kv_cache_size: u64,
    pub deltanet_state_base: u64,
    pub deltanet_state_size: u64,
    pub scratch_base: u64,       // MoE scratch + layer compute temps (M1/M2)
    pub temp_base: u64,          // Layer compute temp (QKV, attn output, etc.)
    pub routing_logits_base: u64,
    pub routing_topk_base: u64,
    pub total_size: u64,
}

const ALIGN: u64 = 64; // cache line size

impl ArenaLayout {
    pub fn compute(cfg: &ModelConfig, weights_size: u64) -> Self {
        let mut off = ALIGN;

        // Weights
        let weights_base = off;
        off += align(weights_size);

        // Hidden states: ping-pong for activations
        let hidden_per_buffer = align(cfg.context_length as u64 * cfg.embedding_length as u64 * 2);
        let hidden_ping = off;
        off += hidden_per_buffer;
        let hidden_pong = off;
        off += hidden_per_buffer;

        // KV cache: Q4_0, (512/32)*18 = 288 B/token for K + 288 B/token for V
        let kv_per_layer = cfg.context_length as u64 * 288 * 2;
        let kv_cache_size = align(10 * kv_per_layer);
        let kv_cache_base = off;
        off += kv_cache_size;

        // DeltaNet state: 30 layers × 16×32×128×128 × 4B
        let deltanet_state_size = align(30 * 16 * 32 * 128u64 * 128 * 4);
        let deltanet_state_base = off;
        off += deltanet_state_size;

        // Snapshot for prefill MoE
        off += align(cfg.context_length as u64 * cfg.embedding_length as u64 * 2);

        // MoE scratch + layer compute temp
        let scratch_base = off;
        off += align(cfg.context_length as u64 * cfg.feed_forward_length as u64 * 2);

        // Layer compute temp: max(hidden*3, 4128) per token × 2B, capped at 8192
        let prefill_batch = (8192u64).min(cfg.context_length as u64);
        let temp_base = off;
        off += align(prefill_batch * (cfg.embedding_length * 3).max(4128) as u64 * 2);

        // Routing logits: seq_len × num_experts × 4B
        let routing_logits_base = off;
        off += align(cfg.context_length as u64 * cfg.expert_count as u64 * 4);

        // GPU topk results: seq_len × 72 bytes (9 entries × 8B each)
        let routing_topk_size = align(cfg.context_length as u64 * 72);
        let routing_topk_base = off;
        off += routing_topk_size;

        ArenaLayout {
            weights_base,
            hidden_ping, hidden_pong,
            kv_cache_base, kv_cache_size,
            deltanet_state_base, deltanet_state_size,
            scratch_base, temp_base, routing_logits_base,
            routing_topk_base,
            total_size: off,
        }
    }
}

fn align(x: u64) -> u64 {
    (x + ALIGN - 1) & !(ALIGN - 1)
}

// ── Tensor Registry ──

/// Name → tensor entry map for O(1) lookup.
#[derive(Default)]
pub struct TensorRegistry {
    pub entries: std::collections::HashMap<String, TensorEntry>,
}

#[derive(Clone)]
pub struct TensorEntry {
    pub arena_offset: u64,
    pub size: u64,
    pub ggml_type: u32,
}

impl TensorRegistry {
    /// Build from parsed GGUF tensors, arena offsets.
    pub fn from_tensors(
        tensors: &[TensorInfo],
        weights_base: u64,
    ) -> Self {
        let by_name: std::collections::HashMap<&str, &TensorInfo> =
            tensors.iter().map(|t| (t.name.as_str(), t)).collect();

        let mut entries = std::collections::HashMap::with_capacity(tensors.len());
        let mut off = weights_base;

        let mut add = |name: &str| {
            if let Some(ti) = by_name.get(name) {
                off = (off + 63) & !63;
                entries.insert(name.to_string(), TensorEntry {
                    arena_offset: off,
                    size: ti.size,
                    ggml_type: ti.ggml_type as u32,
                });
                off += ti.size;
            }
        };

        let max_layer = 40u32;
        let max_exp = 256u32;

        // Group 0: embedding
        add("token_embd.weight");

        // Group 1: per-layer dense
        for l in 0..max_layer {
            for name in &[
                "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate",
                "shared_expert.w1", "shared_expert.w2", "shared_expert.w3",
            ] {
                add(&format!("blk.{}.{}.weight", l, name));
            }
        }

        // Group 2-4: expert weights by layer × expert
        for l in 0..max_layer {
            for e in 0..max_exp {
                add(&format!("blk.{}.experts.{}.w1.weight", l, e));
            }
        }
        for l in 0..max_layer {
            for e in 0..max_exp {
                add(&format!("blk.{}.experts.{}.w2.weight", l, e));
            }
        }
        for l in 0..max_layer {
            for e in 0..max_exp {
                add(&format!("blk.{}.experts.{}.w3.weight", l, e));
            }
        }

        // Group 5: remaining tensors not caught above
        for ti in tensors {
            if !entries.contains_key(&ti.name) {
                off = (off + 63) & !63;
                entries.insert(ti.name.clone(), TensorEntry {
                    arena_offset: off,
                    size: ti.size,
                    ggml_type: ti.ggml_type as u32,
                });
                off += ti.size;
            }
        }

        TensorRegistry { entries }
    }

    pub fn lookup(&self, name: &str) -> Option<&TensorEntry> {
        self.entries.get(name)
    }
}

// ── Layer Weight Index ──

/// Per-layer offsets for all weight matrices.
/// Indexed by layer number (0..block_count).
pub struct LayerWeights {
    pub embedding: u64,          // token_embd.weight
    pub max_seq_len: u32,        // context_length
    pub mtp_w1: Vec<u64>,        // [num_mtp_heads] hidden×hidden
    pub mtp_w2: Vec<u64>,        // [num_mtp_heads] hidden×vocab
    pub num_mtp_heads: u32,      // 2 for Qwen3.6
    pub attn_q: Vec<u64>,
    pub attn_k: Vec<u64>,
    pub attn_v: Vec<u64>,
    pub attn_output: Vec<u64>,
    pub ffn_gate: Vec<u64>,
    pub shared_w1: Vec<u64>,
    pub shared_w2: Vec<u64>,
    pub shared_w3: Vec<u64>,
    // Flat: [layer * expert_count + expert_id]
    pub expert_w1: Vec<u64>,
    pub expert_w2: Vec<u64>,
    pub expert_w3: Vec<u64>,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_layers: u32,
    pub num_experts: u32,
}

impl LayerWeights {
    /// Build from registry by resolving known tensor name patterns.
    pub fn from_registry(
        reg: &TensorRegistry,
        cfg: &ModelConfig,
    ) -> Self {
        let n = cfg.block_count as usize;
        let e = cfg.expert_count as usize;
        let empty = || vec![0u64; n];
        let lk = |name: &str| reg.lookup(name).map(|e| e.arena_offset).unwrap_or(0);

        let mut attn_q = empty();
        let mut attn_k = empty();
        let mut attn_v = empty();
        let mut attn_output = empty();
        let mut ffn_gate = empty();
        let mut shared_w1 = empty();
        let mut shared_w2 = empty();
        let mut shared_w3 = empty();

        for layer in 0..n {
            attn_q[layer] = lk(&format!("blk.{}.attn_q.weight", layer));
            attn_k[layer] = lk(&format!("blk.{}.attn_k.weight", layer));
            attn_v[layer] = lk(&format!("blk.{}.attn_v.weight", layer));
            attn_output[layer] = lk(&format!("blk.{}.attn_output.weight", layer));
            ffn_gate[layer] = lk(&format!("blk.{}.ffn_gate.weight", layer));
            shared_w1[layer] = lk(&format!("blk.{}.shared_expert.w1.weight", layer));
            shared_w2[layer] = lk(&format!("blk.{}.shared_expert.w2.weight", layer));
            shared_w3[layer] = lk(&format!("blk.{}.shared_expert.w3.weight", layer));
        }

        // Expert weights: flat [layer * num_experts + expert]
        let mut expert_w1 = vec![0u64; n * e];
        let mut expert_w2 = vec![0u64; n * e];
        let mut expert_w3 = vec![0u64; n * e];
        for layer in 0..n {
            for expert in 0..e {
                let idx = layer * e + expert;
                expert_w1[idx] = lk(&format!("blk.{}.experts.{}.w1.weight", layer, expert));
                expert_w2[idx] = lk(&format!("blk.{}.experts.{}.w2.weight", layer, expert));
                expert_w3[idx] = lk(&format!("blk.{}.experts.{}.w3.weight", layer, expert));
            }
        }

        // MTP heads (try 2, fall back to 0 if not in model)
        let num_mtp = if lk("output.mtp_head.0.weight") != 0 { 2u32 } else { 0u32 };
        let mut mtp_w1 = Vec::with_capacity(num_mtp as usize);
        let mut mtp_w2 = Vec::with_capacity(num_mtp as usize);
        for i in 0..num_mtp {
            mtp_w1.push(lk(&format!("output.mtp_head.{}.weight", i)));
            mtp_w2.push(lk(&format!("output.mtp_head.{}.output.weight", i)));
        }

        LayerWeights {
            embedding: lk("token_embd.weight"),
            max_seq_len: cfg.context_length,
            mtp_w1, mtp_w2, num_mtp_heads: num_mtp,
            attn_q, attn_k, attn_v, attn_output, ffn_gate,
            shared_w1, shared_w2, shared_w3,
            expert_w1, expert_w2, expert_w3,
            hidden_size: cfg.embedding_length,
            intermediate_size: cfg.feed_forward_length,
            num_layers: cfg.block_count,
            num_experts: cfg.expert_count,
        }
    }
}

// ── Weight Loading ──

/// Copy weights from GGUF tensors to arena using TensorRegistry offsets.
/// Converts F16 tensors to IQ4_XS format on load (GEMM shaders expect IQ4_XS).
pub unsafe fn load_weights_from_tensors<'a>(
    reader: &'a GgufReader<'a>,
    tensors: &[crate::gguf::TensorInfo],
    reg: &TensorRegistry,
    arena_base: *mut u8,
) {
    for ti in tensors {
        if let Some(entry) = reg.lookup(&ti.name) {
            debug_assert_eq!(ti.size, entry.size, "size mismatch for {}", ti.name);
            debug_assert_eq!(ti.ggml_type as u32, entry.ggml_type, "type mismatch for {}", ti.name);
            let src = &reader.data[(reader.tensor_data_offset + ti.offset) as usize
                ..(reader.tensor_data_offset + ti.offset + ti.size) as usize];
            let dst = arena_base.add(entry.arena_offset as usize);
            if ti.ggml_type == crate::gguf::GgmlType::F16 {
                // Convert F16→IQ4_XS so GEMM shaders reading this as IQ4_XS work correctly
                let n = ti.size as usize / 2; // number of f16 values
                let mut iq4xs = vec![0u8; (n + 31) / 32 * 36];
                quantize_f16_to_iq4xs(src, &mut iq4xs, n);
                std::ptr::copy_nonoverlapping(iq4xs.as_ptr(), dst, iq4xs.len());
            } else {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, ti.size as usize);
            }
        }
    }
}

/// Convert F16 bytes to IQ4_XS format. Input is raw f16 little-endian bytes.
pub fn quantize_f16_to_iq4xs(src: &[u8], dst: &mut [u8], n: usize) {
    let blocks = (n + 31) / 32;
    for b in 0..blocks {
        let bo = b * 36;
        let so = b * 64;
        // Find max absolute value in block (up to 32 values, last block may be partial)
        let count = (n - b * 32).min(32);
        let mut max_abs = 0.0f32;
        for i in 0..count {
            let lo = src[so + i*2] as u16;
            let hi = src[so + i*2 + 1] as u16;
            let bits = hi << 8 | lo;
            let val = f16_bits_to_f32(bits);
            let abs = val.abs();
            if abs > max_abs { max_abs = abs; }
        }
        // Compute scale (prevent division by zero)
        let d = if max_abs == 0.0 { 1.0f32 } else { max_abs / 7.0 };
        let d_bits = f32_to_f16_bits(d);
        // Write d and d2 (same scale for both halves)
        dst[bo] = d_bits as u8;
        dst[bo + 1] = (d_bits >> 8) as u8;
        dst[bo + 2] = d_bits as u8;
        dst[bo + 3] = (d_bits >> 8) as u8;
        // Quantize and pack
        let mut nibbles = [0u8; 16];
        let mut qh = [0u8; 16];
        for i in 0..count {
            let lo = src[so + i*2] as u16;
            let hi = src[so + i*2 + 1] as u16;
            let bits = hi << 8 | lo;
            let val = f16_bits_to_f32(bits);
            let q = ((val / d + 8.0).round() as i32).clamp(0, 31) as u32;
            let low4 = q & 0xF;
            let high = (q >> 4) & 1;
            let ni = i >> 1;
            let shift = (i & 1) * 4;
            nibbles[ni] |= (low4 as u8) << shift;
            if high != 0 {
                qh[i >> 3] |= 1 << (i & 7);
            }
        }
        dst[bo + 4..bo + 20].copy_from_slice(&nibbles);
        dst[bo + 20..bo + 36].copy_from_slice(&qh);
    }
}
