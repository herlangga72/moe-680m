use crate::constants::{MTPBlockPC, SamplePC};
use crate::dispatch::{BarrierKind, DispatchChain, DispatchStep};
use crate::engine::Engine;
use crate::error::Result;

/// Multi-Token Prediction (MTP) speculative decoder.
///
/// After the main model generates a token, the MTP runner drafts several
/// future tokens using lightweight MTP transformer heads, then verifies
/// them with a batched forward pass through the full model.
pub struct MtpRunner {
    pub n_modules: u32,
    pub depth: u32,
}

impl MtpRunner {
    pub fn new(n_modules: u32, depth: u32) -> Self {
        Self { n_modules, depth }
    }

    /// Run the MTP draft chain.
    ///
    /// `hidden_state` is the FP32 hidden state from the main model's decode
    /// step. `first_token` is the most recently decoded token (t+1).
    ///
    /// Returns `[t+1, t+2, ..., t+draft_depth]` — the first element is the
    /// already-decoded main-model token, the rest are MTP draft proposals.
    pub fn draft(
        &self,
        engine: &mut Engine,
        hidden_state: &[f32],
        first_token: u32,
    ) -> Result<Vec<u32>> {
        let mut drafts = vec![first_token];
        let mut _mtp_hidden = hidden_state.to_vec();
        let dim = engine.config.hidden_dim;
        let max_d = self.depth.min(self.n_modules);

        for d in 0..max_d {
            let mut chain = DispatchChain::new();

            // ---- MTP concat + norm ----
            // Concatenates the main-model hidden state with the MTP block's
            // token embedding, then applies RMSNorm.
            chain.add(DispatchStep {
                pipeline_name: "mtp_concat_norm",
                push_data: crate::engine::pc_bytes(&MTPBlockPC {
                    dim,
                    head_dim: engine.config.head_dim,
                    n_heads: engine.config.n_heads_q,
                    n_kv_heads: engine.config.n_heads_kv,
                    pos: engine.kv_cache.position() + d + 1,
                    block_idx: d,
                }),
                workgroup_x: crate::engine::div_up(dim, 256),
                workgroup_y: 1,
                workgroup_z: 1,
                buffers: vec![],
                barrier: BarrierKind::ExecOnly,
            });

            // ---- MTP attention (cross-attends to main-model KV cache) ----
            chain.add(DispatchStep {
                pipeline_name: "mtp_attention",
                push_data: crate::engine::pc_bytes(&MTPBlockPC {
                    dim,
                    head_dim: engine.config.head_dim,
                    n_heads: engine.config.n_heads_q,
                    n_kv_heads: engine.config.n_heads_kv,
                    pos: engine.kv_cache.position() + d + 1,
                    block_idx: d,
                }),
                workgroup_x: engine.config.n_heads_q,
                workgroup_y: 1,
                workgroup_z: 1,
                buffers: vec![],
                barrier: BarrierKind::ExecOnly,
            });

            // ---- MTP FFN (SwiGLU: gate + up → silu × mult → down) ----
            chain.add(DispatchStep {
                pipeline_name: "mtp_ffn",
                push_data: crate::engine::pc_bytes(&MTPBlockPC {
                    dim,
                    head_dim: engine.config.head_dim,
                    n_heads: engine.config.n_heads_q,
                    n_kv_heads: engine.config.n_heads_kv,
                    pos: 0,
                    block_idx: d,
                }),
                workgroup_x: crate::engine::div_up(dim, 256),
                workgroup_y: 1,
                workgroup_z: 1,
                buffers: vec![],
                barrier: BarrierKind::ExecOnly,
            });

            // ---- MTP head → sample ----
            // Projects the MTP block's output to vocabulary logits and
            // samples a single token.
            chain.add(DispatchStep {
                pipeline_name: "mtp_head",
                push_data: crate::engine::pc_bytes(&SamplePC {
                    vocab_size: engine.config.vocab_size,
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

            chain.execute(&engine.device, &engine.shaders)?;

            // ponytail: read draft token from mapped token_out buffer
            let draft_token = 0u32; // stub
            drafts.push(draft_token);
        }

        Ok(drafts)
    }

    /// Verify draft tokens through the full model forward pass.
    ///
    /// Runs `engine.verify_forward` on the draft batch, then advances the
    /// KV cache by the number of accepted tokens.
    ///
    /// Returns the number of accepted tokens (minimum 1 — the already-decoded
    /// main-model token `t+1` is always accepted).
    pub fn verify(&self, engine: &mut Engine, drafts: &[u32]) -> Result<u32> {
        let accepted = engine.verify_forward(drafts)?;

        // Advance KV cache past the accepted tokens so the next decode
        // writes at the correct position. Rejected positions will be
        // overwritten naturally on the next iteration.
        for _ in 0..accepted {
            engine.kv_cache.advance()?;
        }

        Ok(accepted)
    }
}
