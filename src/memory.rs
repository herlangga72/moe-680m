// Job 3: Arena allocator (memory layout + Vulkan allocation)
// Job 4: Weight loader (copy GGUF tensors into arena)
// See plans/data-structures.md for struct layouts.

use crate::gguf::{GgufReader, ModelConfig, TensorInfo};
use ash::vk;

// ── Arena Chunking ──

pub struct ArenaChunk {
    pub memory: vk::DeviceMemory,
    pub ptr: *mut u8,
    pub size: u64,
}

pub struct Arena {
    pub chunks: Vec<ArenaChunk>,
    pub base_ptr: *mut u8,
    pub total_size: u64,
}

// ── Arena Layout (pure math, no Vulkan) ──

/// All arena region offsets, computed from ModelConfig.
/// All values in bytes. All aligned to 256 bytes.
pub struct ArenaLayout {
    pub weights_base: u64,
    pub weights_size: u64,
    pub hidden_ping: u64,
    pub hidden_pong: u64,
    pub hidden_size: u64,         // per-buffer
    pub kv_cache_base: u64,
    pub kv_cache_size: u64,
    pub deltanet_state_base: u64,
    pub deltanet_state_size: u64,
    pub snapshot_base: u64,
    pub scratch_base: u64,       // MoE scratch + layer compute temps (M1/M2)
    pub scratch_size: u64,
    pub temp_base: u64,          // Layer compute temp (QKV, attn output, etc.)
    pub temp_size: u64,
    pub routing_logits_base: u64,
    pub routing_logits_size: u64,
    pub total_size: u64,
}

const ALIGN: u64 = 64; // cache line size

impl ArenaLayout {
    pub fn compute(cfg: &ModelConfig, weights_size: u64) -> Self {
        // KV cache needs attention head dim (256), not DeltaNet head dim (128)
        let head_dim_attn = if cfg.architecture.contains("qwen") { 256u64 } else { cfg.head_dim as u64 };
        let kv_heads = cfg.attention_head_count_kv as u64;
        let mut off = ALIGN; // start after offset 0 (null region)

        // Weights
        let weights_base = off;
        let weights_size = align(weights_size);
        off += weights_size;

        // Hidden states: ping-pong for activations
        // Sized for max context (prefill). During generation only 1 token used.
        let hidden_per_buffer = align(cfg.context_length as u64 * cfg.embedding_length as u64 * 2);
        let hidden_ping = off;
        off += hidden_per_buffer;
        let hidden_pong = off;
        off += hidden_per_buffer;

        // KV cache: Q4_0, (512/32)*18 = 288 B/token for K + 288 B/token for V
        let kv_per_layer = cfg.context_length as u64 * 288 * 2;
        let kv_cache_size = align(10 * kv_per_layer); // only 10 attn layers
        let kv_cache_base = off;
        off += kv_cache_size;

        // DeltaNet state: 30 layers × qk_heads × v_heads × head_dim² × 4B
        let deltanet_state_size = align(30 * 16 * 32 * 128u64 * 128 * 4);
        let deltanet_state_base = off;
        off += deltanet_state_size;

        // Snapshot for prefill MoE: seq_len × hidden × 2B
        let snapshot_size = align(cfg.context_length as u64 * cfg.embedding_length as u64 * 2);
        let snapshot_base = off;
        off += snapshot_size;

        // MoE scratch:
        //   gen: 9 experts × 1 token × intermediate × 2B = tiny
        //   prefill: ~18 tokens/exp × 256 exp × 512 × 2B = 4.7 MB
        let scratch_size = align(cfg.context_length as u64 * cfg.feed_forward_length as u64 * 2);
        let scratch_base = off;
        off += scratch_size;

        // Layer compute temp (M1/M2 fix): QKV output or attention output
        // Worst case: max(hidden*3, 4128) per token × 2B = 12 KB/token
        let temp_size = align(cfg.context_length as u64 * (cfg.embedding_length * 3).max(4128) as u64 * 2);
        let temp_base = off;
        off += temp_size;

        // Routing logits: seq_len × num_experts × 4B
        let routing_logits_size = align(cfg.context_length as u64 * cfg.expert_count as u64 * 4);
        let routing_logits_base = off;
        off += routing_logits_size;

        ArenaLayout {
            weights_base,
            weights_size,
            hidden_ping,
            hidden_pong,
            hidden_size: hidden_per_buffer,
            kv_cache_base,
            kv_cache_size,
            deltanet_state_base,
            deltanet_state_size,
            snapshot_base,
            scratch_base,
            scratch_size,
            temp_base,
            temp_size,
            routing_logits_base,
            routing_logits_size,
            total_size: off,
        }
    }
}

fn align(x: u64) -> u64 {
    (x + ALIGN - 1) & !(ALIGN - 1)
}

// ── Tensor Registry ──

/// Sorted array of tensor entries for O(log N) lookup by name hash.
#[derive(Default)]
pub struct TensorRegistry {
    pub entries: Vec<TensorEntry>,
}

#[derive(Clone)]
pub struct TensorEntry {
    pub name_hash: u64,
    pub arena_offset: u64,
    pub size: u64,
    pub ggml_type: u32,
}

impl TensorRegistry {
    /// Build from parsed GGUF tensors in cache-friendly order.
    /// Groups: embedding/dense → per-layer (norms, QKV, output, router, shared)
    /// → expert w1/w2/w3 arrays. All aligned to 64 bytes (cache line).
    pub fn from_tensors(
        tensors: &[TensorInfo],
        weights_base: u64,
    ) -> Self {
        // Build name → TensorInfo map for O(1) lookup
        let by_name: std::collections::HashMap<&str, &TensorInfo> =
            tensors.iter().map(|t| (t.name.as_str(), t)).collect();

        let mut entries: Vec<TensorEntry> = Vec::with_capacity(tensors.len());
        let mut off = weights_base;

        let mut add = |name: &str| {
            if let Some(ti) = by_name.get(name) {
                off = (off + 63) & !63; // 64-byte cache line align
                entries.push(TensorEntry {
                    name_hash: fnv1a(name.as_bytes()),
                    arena_offset: off,
                    size: ti.size,
                    ggml_type: ti.ggml_type as u32,
                });
                off += ti.size;
            }
        };

        let max_layer = 40u32;
        let max_exp = 256u32;

        // Group 0: embedding (accessed every token)
        add("token_embd.weight");

        // Group 1: per-layer dense (40 layers × norms/QKV/output/router/shared)
        for l in 0..max_layer {
            add(&format!("blk.{}.attn_norm.weight", l));
            add(&format!("blk.{}.ffn_norm.weight", l));
            add(&format!("blk.{}.attn_q.weight", l));
            add(&format!("blk.{}.attn_k.weight", l));
            add(&format!("blk.{}.attn_v.weight", l));
            add(&format!("blk.{}.attn_gate.weight", l));
            add(&format!("blk.{}.attn_output.weight", l));
            add(&format!("blk.{}.ffn_gate.weight", l));
            add(&format!("blk.{}.shared_expert.w1.weight", l));
            add(&format!("blk.{}.shared_expert.w2.weight", l));
            add(&format!("blk.{}.shared_expert.w3.weight", l));
        }

        // Group 2: expert w1 array (all layers × experts, contiguous)
        for l in 0..max_layer {
            for e in 0..max_exp {
                add(&format!("blk.{}.experts.{}.w1.weight", l, e));
            }
        }
        // Group 3: expert w2 array
        for l in 0..max_layer {
            for e in 0..max_exp {
                add(&format!("blk.{}.experts.{}.w2.weight", l, e));
            }
        }
        // Group 4: expert w3 array
        for l in 0..max_layer {
            for e in 0..max_exp {
                add(&format!("blk.{}.experts.{}.w3.weight", l, e));
            }
        }

        // Group 5: any remaining tensors not caught above
        // (add closure is dropped with this scope, so we can immutably borrow entries)
        let assigned_hashes: std::collections::HashSet<u64> =
            entries.iter().map(|e| e.name_hash).collect();
        for ti in tensors {
            if !assigned_hashes.contains(&fnv1a(ti.name.as_bytes())) {
                off = (off + 63) & !63;
                entries.push(TensorEntry {
                    name_hash: fnv1a(ti.name.as_bytes()),
                    arena_offset: off,
                    size: ti.size,
                    ggml_type: ti.ggml_type as u32,
                });
                off += ti.size;
            }
        }

        // Sort by hash for binary search
        entries.sort_by_key(|e| e.name_hash);

        TensorRegistry { entries }
    }

    /// Look up a tensor by name. Returns None if not found.
    pub fn lookup(&self, name: &str) -> Option<&TensorEntry> {
        let hash = fnv1a(name.as_bytes());
        self.entries
            .binary_search_by_key(&hash, |e| e.name_hash)
            .ok()
            .map(|i| &self.entries[i])
    }
}

// FNV-1a hash (64-bit)
fn fnv1a(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ── Layer Weight Index ──

/// Per-layer offsets for all weight matrices.
/// Indexed by layer number (0..block_count).
pub struct LayerWeights {
    pub embedding: u64,          // token_embd.weight (for H6)
    pub max_seq_len: u32,        // context_length
    pub attn_q: Vec<u64>,
    pub attn_k: Vec<u64>,
    pub attn_v: Vec<u64>,
    pub attn_output: Vec<u64>,
    pub attn_gate: Vec<u64>,
    pub attn_norm: Vec<u64>,
    pub ffn_norm: Vec<u64>,
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
        let mut empty = || vec![0u64; n];
        let lk = |name: &str| reg.lookup(name).map(|e| e.arena_offset).unwrap_or(0);

        let mut attn_q = empty();
        let mut attn_k = empty();
        let mut attn_v = empty();
        let mut attn_output = empty();
        let mut attn_gate = empty();
        let mut attn_norm = empty();
        let mut ffn_norm = empty();
        let mut ffn_gate = empty();
        let mut shared_w1 = empty();
        let mut shared_w2 = empty();
        let mut shared_w3 = empty();

        for layer in 0..n {
            attn_q[layer] = lk(&format!("blk.{}.attn_q.weight", layer));
            attn_k[layer] = lk(&format!("blk.{}.attn_k.weight", layer));
            attn_v[layer] = lk(&format!("blk.{}.attn_v.weight", layer));
            attn_output[layer] = lk(&format!("blk.{}.attn_output.weight", layer));
            attn_gate[layer] = lk(&format!("blk.{}.attn_gate.weight", layer));
            attn_norm[layer] = lk(&format!("blk.{}.attn_norm.weight", layer));
            ffn_norm[layer] = lk(&format!("blk.{}.ffn_norm.weight", layer));
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

        LayerWeights {
            embedding: lk("token_embd.weight"),
            max_seq_len: cfg.context_length,
            attn_q, attn_k, attn_v, attn_output, attn_gate,
            attn_norm, ffn_norm, ffn_gate,
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
pub unsafe fn load_weights_from_tensors(
    reader: &GgufReader,
    tensors: &[crate::gguf::TensorInfo],
    reg: &TensorRegistry,
    arena_base: *mut u8,
) {
    for ti in tensors {
        if let Some(entry) = reg.lookup(&ti.name) {
            let src = &reader.data[(reader.tensor_data_offset + ti.offset) as usize
                ..(reader.tensor_data_offset + ti.offset + ti.size) as usize];
            let dst = arena_base.add(entry.arena_offset as usize);
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, ti.size as usize);
        }
    }
}
