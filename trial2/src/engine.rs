use ash::vk;
use crate::constants::{AttentionPC, LinearPC, MoEPC, RMSNormPC, RouterPC, SamplePC};
use crate::device::Device;
use crate::dispatch::{BarrierKind, DispatchChain, DispatchStep};
use crate::error::Result;
use crate::gguf::{GgufFile, ModelConfig};
use crate::kv_cache::KvCache;
use crate::memory::{Arena, Buffer};
use crate::shaders::ShaderCache;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Ceiling integer division.
pub fn div_up(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

/// Serialise a `Pod` push-constant struct into a 128-byte array.
/// Shorter structs are zero-padded.
pub(crate) fn pc_bytes<T: bytemuck::Pod>(pc: &T) -> [u8; 128] {
    let mut data = [0u8; 128];
    let bytes = bytemuck::bytes_of(pc);
    let len = bytes.len().min(128);
    data[..len].copy_from_slice(&bytes[..len]);
    data
}

// ---------------------------------------------------------------------------
// EngineBuffers
// ---------------------------------------------------------------------------

pub struct EngineBuffers {
    pub hidden_state: Buffer,
    pub hidden_fp32: Buffer,
    pub q: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub attn_output: Buffer,
    pub gate_logits: Buffer,
    pub moe_intermediate: Buffer,
    pub moe_output: Buffer,
    pub logits: Buffer,
    pub token_out: Buffer,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct Engine {
    pub config: ModelConfig,
    pub kv_cache: KvCache,
    pub buffers: EngineBuffers,
    pub(crate) device: Device,
    pub(crate) shaders: ShaderCache,
}

impl Engine {
    /// Allocate all compute buffers and the KV cache.
    ///
    /// The caller is responsible for sizing `arena` large enough to hold
    /// every buffer plus the KV cache.
    pub fn new(
        gguf: &GgufFile,
        mut arena: Arena,
        device: Device,
        shaders: ShaderCache,
        max_context: u32,
    ) -> Result<Self> {
        let config = gguf.model_config()?;
        let dim = config.hidden_dim as u64;
        let n_heads = config.n_heads_q as u64;
        let n_kv = config.n_heads_kv as u64;
        let head_dim = config.head_dim as u64;
        let n_active = config.n_active_experts as u64;
        let ffn = config.ffn_intermediate as u64;
        let vocab = config.vocab_size as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;

        // Create buffer handles (memory not yet bound)
        let hidden_fp32 = Buffer::new(&device.device, dim * 4, usage)?;
        let hidden_fp16 = Buffer::new(&device.device, dim * 2, usage)?;
        let q = Buffer::new(&device.device, n_heads * head_dim * 2, usage)?;
        let k = Buffer::new(&device.device, n_kv * head_dim * 2, usage)?;
        let v = Buffer::new(&device.device, n_kv * head_dim * 2, usage)?;
        let attn_out = Buffer::new(&device.device, dim * 2, usage)?;
        let gate = Buffer::new(&device.device, config.n_experts as u64 * 2, usage)?;
        let moe_int = Buffer::new(&device.device, n_active * ffn * 4, usage)?;
        let moe_out = Buffer::new(&device.device, n_active * dim * 2, usage)?;
        let logits = Buffer::new(&device.device, vocab * 4, usage)?;
        let token = Buffer::new(&device.device, 4, usage)?;

        // Allocate arena space and bind each buffer
        arena.allocate("hidden_fp32", dim * 4)?;
        arena.bind_buffer("hidden_fp32", &hidden_fp32)?;
        arena.allocate("hidden_state", dim * 2)?;
        arena.bind_buffer("hidden_state", &hidden_fp16)?;
        arena.allocate("q", n_heads * head_dim * 2)?;
        arena.bind_buffer("q", &q)?;
        arena.allocate("k", n_kv * head_dim * 2)?;
        arena.bind_buffer("k", &k)?;
        arena.allocate("v", n_kv * head_dim * 2)?;
        arena.bind_buffer("v", &v)?;
        arena.allocate("attn_output", dim * 2)?;
        arena.bind_buffer("attn_output", &attn_out)?;
        arena.allocate("gate_logits", config.n_experts as u64 * 2)?;
        arena.bind_buffer("gate_logits", &gate)?;
        arena.allocate("moe_intermediate", n_active * ffn * 4)?;
        arena.bind_buffer("moe_intermediate", &moe_int)?;
        arena.allocate("moe_output", n_active * dim * 2)?;
        arena.bind_buffer("moe_output", &moe_out)?;
        arena.allocate("logits", vocab * 4)?;
        arena.bind_buffer("logits", &logits)?;
        arena.allocate("token_out", 4)?;
        arena.bind_buffer("token_out", &token)?;

        // Create the KV cache (allocates and binds its own arena entries)
        let kv_cache = KvCache::new(&config, max_context, &mut arena, &device.device)?;

        Ok(Self {
            config,
            kv_cache,
            buffers: EngineBuffers {
                hidden_state: hidden_fp16,
                hidden_fp32: hidden_fp32,
                q,
                k,
                v,
                attn_output: attn_out,
                gate_logits: gate,
                moe_intermediate: moe_int,
                moe_output: moe_out,
                logits,
                token_out: token,
            },
            device,
            shaders,
        })
    }

    // -----------------------------------------------------------------------
    // Layer dispatch chain
    // -----------------------------------------------------------------------

    /// Append every compute dispatch for one transformer layer to `chain`.
    ///
    /// `seq_len` is the number of tokens being processed (batch size during
    /// prefill, 1 during decode).
    fn build_layer_chain(
        &self,
        chain: &mut DispatchChain,
        layer: u32,
        seq_len: u32,
        _is_prefill: bool,
    ) {
        let dim = self.config.hidden_dim;
        let n_heads = self.config.n_heads_q;
        let n_kv = self.config.n_heads_kv;
        let n_active = self.config.n_active_experts;
        let head_dim = self.config.head_dim;
        let full_attn_interval = 4u32; // every 4th layer is full attention
        let is_full_attn = (layer % full_attn_interval) == 0;

        // ── SSM branch (all layers have SSM weights) ──
        {
            // SSM pre-norm (group RMS, 128-element weight tiled 16×)
            chain.add(DispatchStep {
                pipeline_name: "ssm_norm",
                push_data: pc_bytes(&RMSNormPC { rows: 128, dim, eps: self.config.eps }),
                workgroup_x: 1, workgroup_y: 1, workgroup_z: 1,
                buffers: vec![],
                barrier: BarrierKind::ExecOnly,
            });

            // Alpha/beta projection (hidden → alpha[32], beta[32])
            chain.add(DispatchStep {
                pipeline_name: "ssm_proj",
                push_data: pc_bytes(&RMSNormPC { rows: 32, dim, eps: 0.0 }),
                workgroup_x: 1, workgroup_y: 1, workgroup_z: 1,
                buffers: vec![],
                barrier: BarrierKind::ExecOnly,
            });

            // Conv1d (kernel=4, 8192 channels)
            chain.add(DispatchStep {
                pipeline_name: "ssm_conv",
                push_data: pc_bytes(&RMSNormPC { rows: 4, dim: 8192, eps: 0.0 }),
                workgroup_x: div_up(8192, 256), workgroup_y: 1, workgroup_z: 1,
                buffers: vec![],
                barrier: BarrierKind::ExecOnly,
            });

            // Selective scan (16 groups, 32 state dim, 256 ch/group)
            chain.add(DispatchStep {
                pipeline_name: "ssm_scan",
                push_data: pc_bytes(&RMSNormPC { rows: 32, dim: 16, eps: 256.0 }),
                workgroup_x: 16, workgroup_y: 1, workgroup_z: 1,
                buffers: vec![],
                barrier: BarrierKind::ExecOnly,
            });

            // Output projection (4096 → 2048)
            chain.add(DispatchStep {
                pipeline_name: "ssm_out",
                push_data: pc_bytes(&RMSNormPC { rows: dim, dim: 4096, eps: 0.0 }),
                workgroup_x: div_up(dim, 256), workgroup_y: 1, workgroup_z: 1,
                buffers: vec![],
                barrier: BarrierKind::ExecOnly,
            });
        }

        // ── Attention branch ──
        if is_full_attn {
            // Full attention (every 4th layer): Q/K separate projections + QK norms
            // ponytail: use existing qkv/attention shaders for now; full-attn variant TBD
        }

        // ---- 1. RMSNorm (pre-attention) ----
        chain.add(DispatchStep {
            pipeline_name: "rms_norm",
            push_data: pc_bytes(&RMSNormPC {
                rows: seq_len,
                dim,
                eps: self.config.eps,
            }),
            workgroup_x: div_up(dim, 256),
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // ---- 2. QKV projection ----
        chain.add(DispatchStep {
            pipeline_name: "qkv",
            push_data: pc_bytes(&LinearPC {
                in_dim: dim,
                out_dim: n_heads * head_dim,
                pad: [0; 2],
            }),
            workgroup_x: div_up(dim, 64),
            workgroup_y: n_heads,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // ---- 3. RoPE ----
        chain.add(DispatchStep {
            pipeline_name: "rope",
            push_data: pc_bytes(&AttentionPC {
                seq_len,
                n_heads,
                n_kv_heads: n_kv,
                head_dim,
                max_seq_len: 0,
            }),
            workgroup_x: n_heads,
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // ---- 4. Attention ----
        chain.add(DispatchStep {
            pipeline_name: "attention",
            push_data: pc_bytes(&AttentionPC {
                seq_len,
                n_heads,
                n_kv_heads: n_kv,
                head_dim,
                max_seq_len: self.kv_cache.max_seq,
            }),
            workgroup_x: n_heads,
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // ---- 5. KV write (MemoryFlush — next token needs to see KV) ----
        chain.add(DispatchStep {
            pipeline_name: "kv_write",
            push_data: pc_bytes(&AttentionPC {
                seq_len,
                n_heads,
                n_kv_heads: n_kv,
                head_dim,
                max_seq_len: self.kv_cache.max_seq,
            }),
            workgroup_x: n_kv,
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::MemoryFlush,
        });

        // ---- 6. Residual add (attn_output + hidden) ----
        chain.add(DispatchStep {
            pipeline_name: "residual_add",
            push_data: pc_bytes(&RMSNormPC {
                rows: seq_len,
                dim,
                eps: 0.0,
            }),
            workgroup_x: div_up(dim, 256),
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // ---- 7. RMSNorm (pre-MoE) ----
        chain.add(DispatchStep {
            pipeline_name: "rms_norm",
            push_data: pc_bytes(&RMSNormPC {
                rows: seq_len,
                dim,
                eps: self.config.eps,
            }),
            workgroup_x: div_up(dim, 256),
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // ---- 8. Router ----
        chain.add(DispatchStep {
            pipeline_name: "router_topk",
            push_data: pc_bytes(&RouterPC {
                dim,
                n_experts: self.config.n_experts,
                n_active,
                n_shared: self.config.n_shared_experts,
            }),
            workgroup_x: 1,
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // ---- 9. MoE gate + up (one dispatch per active expert) ----
        for e in 0..n_active {
            chain.add(DispatchStep {
                pipeline_name: "moe_gate_up",
                push_data: pc_bytes(&MoEPC {
                    dim,
                    intermediate: self.config.ffn_intermediate,
                    expert_idx: e,
                    is_shared: 0,
                }),
                workgroup_x: div_up(dim, 64),
                workgroup_y: 1,
                workgroup_z: 1,
                buffers: vec![],
                barrier: BarrierKind::ExecOnly,
            });
        }

        // ---- 10. SiLU multiply (all experts in parallel) ----
        chain.add(DispatchStep {
            pipeline_name: "silu_mult",
            push_data: pc_bytes(&MoEPC {
                dim,
                intermediate: self.config.ffn_intermediate,
                expert_idx: 0,
                is_shared: 0,
            }),
            workgroup_x: div_up(self.config.ffn_intermediate, 256),
            workgroup_y: n_active,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // ---- 11. MoE down (one dispatch per active expert) ----
        for e in 0..n_active {
            chain.add(DispatchStep {
                pipeline_name: "moe_down",
                push_data: pc_bytes(&MoEPC {
                    dim,
                    intermediate: self.config.ffn_intermediate,
                    expert_idx: e,
                    is_shared: 0,
                }),
                workgroup_x: div_up(dim, 64),
                workgroup_y: 1,
                workgroup_z: 1,
                buffers: vec![],
                barrier: BarrierKind::ExecOnly,
            });
        }

        // ---- 12. MoE combine (weighted sum of expert outputs) ----
        chain.add(DispatchStep {
            pipeline_name: "moe_combine",
            push_data: pc_bytes(&RouterPC {
                dim,
                n_experts: self.config.n_experts,
                n_active,
                n_shared: 0,
            }),
            workgroup_x: div_up(dim, 256),
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // ---- 13. Residual add (moe_output + hidden) ----
        chain.add(DispatchStep {
            pipeline_name: "residual_add",
            push_data: pc_bytes(&RMSNormPC {
                rows: seq_len,
                dim,
                eps: 0.0,
            }),
            workgroup_x: div_up(dim, 256),
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });
    }

    // -----------------------------------------------------------------------
    // Prefill (batch prompt processing)
    // -----------------------------------------------------------------------

    /// Embed all prompt tokens, run every transformer layer, and sample the
    /// next token. The KV cache is advanced by `tokens.len()`.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        let seq_len = tokens.len() as u32;
        let mut chain = DispatchChain::new();

        // Embed all tokens (batch)
        // ponytail: embed shader uses seq_len from push constants
        chain.add(DispatchStep {
            pipeline_name: "embed",
            push_data: pc_bytes(&AttentionPC {
                seq_len,
                n_heads: 0,
                n_kv_heads: 0,
                head_dim: 0,
                max_seq_len: 0,
            }),
            workgroup_x: seq_len,
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // All transformer layers
        for layer in 0..self.config.n_layers {
            self.build_layer_chain(&mut chain, layer, seq_len, true);
        }

        // Advance KV cache by the full batch
        for _ in 0..seq_len {
            self.kv_cache.advance()?;
        }

        // LM head: project last-token hidden state to vocabulary logits
        chain.add(DispatchStep {
            pipeline_name: "lm_head",
            push_data: pc_bytes(&LinearPC {
                in_dim: self.config.hidden_dim,
                out_dim: self.config.vocab_size,
                pad: [0; 2],
            }),
            workgroup_x: div_up(self.config.vocab_size, 256),
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // Sample: argmax or top-p/top-k from logits → token_out
        chain.add(DispatchStep {
            pipeline_name: "sample",
            push_data: pc_bytes(&SamplePC {
                vocab_size: self.config.vocab_size,
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
            }),
            workgroup_x: 1,
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::HostRead,
        });

        chain.execute(&self.device, &self.shaders)?;

        // ponytail: read mapped token_out buffer
        // Real impl: memcpy from mapped arena + device wait + synchronisation
        let token = 0u32;
        Ok(token)
    }

    // -----------------------------------------------------------------------
    // Decode (single-token generation)
    // -----------------------------------------------------------------------

    /// Embed one token, run all layers, sample. Returns the sampled token
    /// and a copy of the final hidden state (FP32) for use by MTP drafting.
    pub fn decode(&mut self, _token: u32) -> Result<(u32, Vec<f32>)> {
        let mut chain = DispatchChain::new();

        // Embed single token
        chain.add(DispatchStep {
            pipeline_name: "embed",
            push_data: pc_bytes(&AttentionPC {
                seq_len: 1,
                n_heads: 0,
                n_kv_heads: 0,
                head_dim: 0,
                max_seq_len: 0,
            }),
            workgroup_x: 1,
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // All layers (single-token path)
        for layer in 0..self.config.n_layers {
            self.build_layer_chain(&mut chain, layer, 1, false);
        }

        // Advance KV cache
        self.kv_cache.advance()?;

        // LM head
        chain.add(DispatchStep {
            pipeline_name: "lm_head",
            push_data: pc_bytes(&LinearPC {
                in_dim: self.config.hidden_dim,
                out_dim: self.config.vocab_size,
                pad: [0; 2],
            }),
            workgroup_x: div_up(self.config.vocab_size, 256),
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // Sample
        chain.add(DispatchStep {
            pipeline_name: "sample",
            push_data: pc_bytes(&SamplePC {
                vocab_size: self.config.vocab_size,
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
            }),
            workgroup_x: 1,
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::HostRead,
        });

        chain.execute(&self.device, &self.shaders)?;

        // ponytail: read mapped buffers and sync
        let sampled = 0u32;
        let hidden = vec![0.0f32; self.config.hidden_dim as usize];
        Ok((sampled, hidden))
    }

    // -----------------------------------------------------------------------
    // Verify (speculative-decoding verification pass)
    // -----------------------------------------------------------------------

    /// Run a full forward pass on `drafts` as a batch and return the number
    /// of consecutively accepted draft tokens (minimum 1, since `drafts[0]`
    /// is the already-verified main-model token).
    ///
    /// The KV cache is NOT advanced by this method; the caller (e.g. MTP)
    /// must advance by the returned count.
    pub fn verify_forward(&mut self, drafts: &[u32]) -> Result<u32> {
        if drafts.is_empty() {
            return Ok(0);
        }
        let seq_len = drafts.len() as u32;
        let mut chain = DispatchChain::new();

        // Embed all draft tokens
        chain.add(DispatchStep {
            pipeline_name: "embed",
            push_data: pc_bytes(&AttentionPC {
                seq_len,
                n_heads: 0,
                n_kv_heads: 0,
                head_dim: 0,
                max_seq_len: 0,
            }),
            workgroup_x: seq_len,
            workgroup_y: 1,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        // All layers (batch mode — reads from current KV cache position,
        // writes KV entries starting at the current position)
        for layer in 0..self.config.n_layers {
            self.build_layer_chain(&mut chain, layer, seq_len, true);
        }

        // LM head over the full sequence (each position predicts next token)
        chain.add(DispatchStep {
            pipeline_name: "lm_head",
            push_data: pc_bytes(&LinearPC {
                in_dim: self.config.hidden_dim,
                out_dim: self.config.vocab_size,
                pad: [0; 2],
            }),
            workgroup_x: div_up(self.config.vocab_size, 256),
            workgroup_y: seq_len,
            workgroup_z: 1,
            buffers: vec![],
            barrier: BarrierKind::ExecOnly,
        });

        chain.execute(&self.device, &self.shaders)?;

        // ponytail: read logits buffer, compute argmax at each position,
        // compare against the next draft token.
        //
        // Position i predicts the token at position i+1.
        // drafts[0] is the main-model token (always accepted).
        // We compare model predictions against drafts[1..].
        //
        // Stub: accept the first token only.
        let accepted = if seq_len >= 2 { 1u32 } else { seq_len };
        Ok(accepted)
    }
}
