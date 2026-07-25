// Jobs 7-10: Inference engine — Vulkan command buffer recording and dispatch.
// Fixed: C1-C4 (descriptor binding, dispatch dims, router dispatch)

use crate::device::DeviceContext;
use crate::memory::{ArenaLayout, LayerWeights};
use crate::pipeline::{PipelineResources, PipelineType};
use crate::router;
use ash::vk;

// ── Push constants (matches GLSL, 128 bytes) ──

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub input_offset: u64,
    pub weights_offset: u64,
    pub output_offset: u64,
    pub M: u32,
    pub N: u32,
    pub K: u32,
    pub num_experts: u32,
    pub routing_weights_off: u64,
    pub num_qk_heads: u32,
    pub num_v_heads: u32,
    pub layer_idx: u32,
    _pad1: u32,
    pub token_ids_off: u64,
    pub scale_factor: f32,
    _pad: [u8; 52],
}

unsafe impl bytemuck::Zeroable for PushConstants {}
unsafe impl bytemuck::Pod for PushConstants {}

impl Default for PushConstants {
    fn default() -> Self {
        Self {
            input_offset: 0, weights_offset: 0, output_offset: 0,
            M: 1, N: 0, K: 0,
            num_experts: 0, routing_weights_off: 0,
            num_qk_heads: 16, num_v_heads: 32,
            layer_idx: 0, _pad1: 0,
            token_ids_off: 0, scale_factor: 1.0,
            _pad: [0u8; 52],
        }
    }
}

// ── Inference state ──

pub struct InferenceState {
    pub position: u32,
    pub seq_len: u32,
    pub kv_cache_filled: u32,
    pub hidden_ping: bool,
}

impl InferenceState {
    pub fn new() -> Self {
        Self { position: 0, seq_len: 0, kv_cache_filled: 0, hidden_ping: true }
    }
}

// ── Engine ──

pub struct InferenceEngine {
    pub device: DeviceContext,
    pub pipelines: PipelineResources,
    pub layout: ArenaLayout,
    pub weights: LayerWeights,
    pub arena_base: *mut u8,
    pub cmd_layer: vk::CommandBuffer,
    pub cmd_moe: vk::CommandBuffer,
    pub fence: vk::Fence,
}

impl InferenceEngine {
    pub fn new(
        device: DeviceContext,
        pipelines: PipelineResources,
        layout: ArenaLayout,
        weights: LayerWeights,
        arena_base: *mut u8,
    ) -> Result<Self, String> {
        let pool_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: device.queue_family,
            ..Default::default()
        };
        let cmd_pool = unsafe {
            device.device.create_command_pool(&pool_info, None)
                .map_err(|e| format!("Cmd pool: {}", e))?
        };
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 2,
            ..Default::default()
        };
        let bufs = unsafe {
            device.device.allocate_command_buffers(&alloc_info)
                .map_err(|e| format!("Alloc cmd bufs: {}", e))?
        };
        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        let fence = unsafe {
            device.device.create_fence(&fence_info, None)
                .map_err(|e| format!("Fence: {}", e))?
        };
        Ok(InferenceEngine {
            device, pipelines, layout, weights, arena_base,
            cmd_layer: bufs[0], cmd_moe: bufs[1], fence,
        })
    }

    // ── Helpers ──

    fn hi(&self, ping: bool) -> u64 { if ping { self.layout.hidden_ping } else { self.layout.hidden_pong } }
    fn ho(&self, ping: bool) -> u64 { if ping { self.layout.hidden_pong } else { self.layout.hidden_ping } }
    fn scratch(&self) -> u64 { self.layout.scratch_base }
    fn temp(&self) -> u64 { self.layout.temp_base }
    fn div64(&self, x: u32) -> u32 { (x + 63) / 64 }   // M: TG covers 64
    fn div8(&self, x: u32) -> u32 { (x + 7) / 8 }      // N: TG covers 8 (RDNA2)
    fn div256(&self, x: u32) -> u32 { (x + 255) / 256 } // RMS norm: 256 threads

    fn push(&self, cmd: vk::CommandBuffer, pc: &PushConstants) {
        unsafe {
            self.device.device.cmd_push_constants(cmd,
                self.pipelines.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(pc));
        }
    }

    fn bind_pipe(&self, cmd: vk::CommandBuffer, pt: PipelineType) {
        unsafe {
            self.device.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE,
                self.pipelines.pipelines[pt as usize]);
            self.device.device.cmd_bind_descriptor_sets(cmd,
                vk::PipelineBindPoint::COMPUTE, self.pipelines.pipeline_layout,
                0, &[self.pipelines.desc_set], &[]);
        }
    }

    fn barrier(&self, cmd: vk::CommandBuffer) {
        let b = vk::MemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            ..Default::default()
        };
        unsafe {
            self.device.device.cmd_pipeline_barrier(cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[b], &[], &[]);
        }
    }

    fn submit(&self, cmd: vk::CommandBuffer) -> Result<(), String> {
        let si = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        unsafe {
            self.device.device.queue_submit(self.device.queue, &[si], self.fence)
                .map_err(|e| format!("Submit: {}", e))?;
            self.device.device.wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("Wait: {}", e))?;
        }
        Ok(())
    }

    // ── Public API ──

    /// Copy a token's embedding from the embedding table to the hidden input buffer.
    /// Must be called before each forward pass (H6: autoregressive token feeding).
    pub fn embed_token(&self, token_id: u32, pos: u32) {
        let embed_size = self.weights.hidden_size as usize * 2; // f16
        let src = unsafe { self.arena_base.add(self.weights.embedding as usize
            + token_id as usize * embed_size) };
        let dst = unsafe { self.arena_base.add(self.layout.hidden_ping as usize
            + pos as usize * embed_size) };
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, embed_size);
        }
    }

    pub fn generate(&mut self, tokens: &[u32], state: &mut InferenceState) -> Result<u32, String> {
        if state.seq_len == 0 {
            self.prefill(tokens, state)?;
            // After prefill, the last token's hidden state is in the output buffer.
            // The forward pass will use it as input for the first generation token.
        }
        let token = self.forward_single(state)?;
        // (C7) Embed token for next generation step
        self.embed_token(token, state.seq_len);
        state.seq_len += 1;
        state.position += 1;
        Ok(token)
    }

    // ── Prefill (M5: embed prompt tokens first) ──

    fn prefill(&mut self, tokens: &[u32], state: &mut InferenceState) -> Result<(), String> {
        let M = tokens.len() as u32;

        // (M5) Embed all prompt tokens into hidden_ping
        for (i, &tid) in tokens.iter().enumerate() {
            self.embed_token(tid, i as u32);
        }

        state.seq_len = M;
        state.hidden_ping = true;
        for layer in 0..self.weights.num_layers {
            let is_gqa = layer % 4 == 3;
            self.record_and_submit_layer(layer, is_gqa, M, state)?;

            // APU bus relief during prefill: every 8 layers, yield to OS
            // Prevents GPU bus saturation from starving UI/system memory requests
            if layer % 8 == 7 && M > 128 {
                std::thread::yield_now();
            }

            // Prefill routing still happens per-layer for correctness
            self.record_and_submit_router(layer, M)?;
            let routing = self.do_route(M);
            self.record_and_submit_moe(layer, &routing, M)?;
            state.hidden_ping = !state.hidden_ping;
        }
        state.kv_cache_filled = M;
        Ok(())
    }

    fn forward_single(&mut self, state: &mut InferenceState) -> Result<u32, String> {
        let M = 1;
        for layer in 0..self.weights.num_layers {
            let is_gqa = layer % 4 == 3;
            self.record_and_submit_layer(layer, is_gqa, M, state)?;
            self.record_and_submit_router(layer, M)?;
            let routing = self.do_route(M);
            self.record_and_submit_moe(layer, &routing, M)?;
            state.hidden_ping = !state.hidden_ping;
        }

        // (C6) lm_head projection: hidden_state × embedding → logits
        // Output hidden state is at ho (after the last layer's ping-pong toggle)
        let ho = self.ho(!state.hidden_ping); // current output
        let scratch = self.scratch();
        let cmd = self.cmd_moe; // reuse moe cmd buffer for lm_head
        let dev = &self.device.device;
        unsafe {
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        }
        let begin = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe { dev.begin_command_buffer(cmd, &begin).unwrap(); }

        // lm_head[tok][vocab] = hidden[tok][hidden] × emb[hidden][vocab]^T
        // Use attn_output shader (standard GEMM)
        self.bind_pipe(cmd, PipelineType::AttnOutput);
        let mut pc = PushConstants::default();
        pc.input_offset = ho;            // hidden state (1 × 2048 f16)
        pc.weights_offset = self.weights.embedding; // token_embd.weight
        pc.output_offset = scratch;       // logits (1 × vocab f16)
        pc.M = 1; pc.N = 248320; pc.K = self.weights.hidden_size;
        self.push(cmd, &pc);
        unsafe { dev.cmd_dispatch(cmd, self.div64(1), self.div8(248320), 1); }

        unsafe { dev.end_command_buffer(cmd).unwrap(); }
        self.submit(cmd)?;

        // Read logits from scratch as f16, convert to f32, sample
        let logits_f16 = unsafe {
            std::slice::from_raw_parts(
                self.arena_base.add(scratch as usize) as *const u16, 248320)
        };
        let mut logits = vec![0.0f32; 248320];
        for (i, &f16_bits) in logits_f16.iter().enumerate() {
            let sign = (f16_bits & 0x8000) as u32;
            let exp = ((f16_bits & 0x7C00) >> 10) as u32;
            let mant = (f16_bits & 0x03FF) as u32;
            let f32_bits = if exp == 0 { sign }
                else if exp == 31 { sign | 0x7F800000 | (mant << 13) }
                else { sign | ((exp + 112) << 23) | (mant << 13) };
            logits[i] = f32::from_bits(f32_bits);
        }
        let token = crate::router::argmax(&logits);
        Ok(token)
    }

    // ── Layer recording (M1: residual to ho, M2: use temp, M3: kv position) ──

    fn record_and_submit_layer(&self, layer: u32, is_gqa: bool, M: u32, s: &InferenceState)
        -> Result<(), String> {
        let cmd = self.cmd_layer;
        let dev = &self.device.device;
        unsafe {
            dev.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            dev.reset_fences(&[self.fence]).unwrap();
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        }
        let begin = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe { dev.begin_command_buffer(cmd, &begin).unwrap(); }

        let hi = self.hi(s.hidden_ping);      // input buffer
        let ho = self.ho(s.hidden_ping);       // output buffer
        let tmp = self.temp();                 // temp for QKV/attention output (M2)
        let hidden = self.weights.hidden_size;
        let out_off = tmp + M as u64 * hidden as u64 * 2; // GEMM output after QKV

        // ── RMS Norm: hi → tmp (norm_out) ──
        self.bind_pipe(cmd, PipelineType::RmsNorm);
        let mut pc = PushConstants::default();
        pc.input_offset = hi;
        pc.output_offset = tmp;
        pc.M = M; pc.K = hidden;
        self.push(cmd, &pc);
        unsafe { dev.cmd_dispatch(cmd, M, 1, 1); }
        self.barrier(cmd);

        if is_gqa {
            // ── Q projection: hidden × attn_q → tmp (Q output) ──
            self.bind_pipe(cmd, PipelineType::GqaQkv);
            pc.input_offset = tmp;
            pc.weights_offset = self.weights.attn_q[layer as usize];
            pc.output_offset = tmp + M as u64 * hidden as u64 * 2;
            pc.M = M; pc.N = 4096; pc.K = hidden; // 16 Q heads × 256 dim
            self.push(cmd, &pc);
            unsafe { dev.cmd_dispatch(cmd, self.div64(M), self.div8(4096), 1); }
            self.barrier(cmd);

            // ── K projection: hidden × attn_k → tmp + hidden*2 + 4096 ──
            self.bind_pipe(cmd, PipelineType::GqaQkv);
            pc.input_offset = tmp;
            pc.weights_offset = self.weights.attn_k[layer as usize];
            pc.output_offset = tmp + M as u64 * hidden as u64 * 2 + 4096;
            pc.M = M; pc.N = 512; pc.K = hidden; // 2 KV heads × 256 dim
            self.push(cmd, &pc);
            unsafe { dev.cmd_dispatch(cmd, self.div64(M), self.div8(512), 1); }
            self.barrier(cmd);

            // ── V projection: hidden × attn_v → tmp + hidden*2 + 4096 + 512 ──
            self.bind_pipe(cmd, PipelineType::GqaQkv);
            pc.input_offset = tmp;
            pc.weights_offset = self.weights.attn_v[layer as usize];
            pc.output_offset = tmp + M as u64 * hidden as u64 * 2 + 4096 + 512;
            pc.M = M; pc.N = 512; pc.K = hidden;
            self.push(cmd, &pc);
            unsafe { dev.cmd_dispatch(cmd, self.div64(M), self.div8(512), 1); }
            self.barrier(cmd);

            // ── (H4/M3) KV cache write at position s.seq_len ──
            // GQA layers are at indices 3,7,11,...,39 → compact index = layer/4
            let gqa_idx = layer / 4;
            let cache_layer = self.layout.kv_cache_base
                + gqa_idx as u64 * (self.layout.kv_cache_size / 10);
            self.bind_pipe(cmd, PipelineType::KvWrite);
            pc.input_offset = tmp + M as u64 * hidden as u64 * 2; // Q+K+V buffer
            pc.weights_offset = cache_layer;
            pc.M = M; pc.K = s.seq_len;            // write position
            pc.N = self.weights.max_seq_len;        // max seq len (for V offset)
            self.push(cmd, &pc);
            unsafe { dev.cmd_dispatch(cmd, self.div64(M), 32, 1); } // 16 K + 16 V Q4_0 blocks
            self.barrier(cmd);

            // ── GQA Attention: Q from tmp+hidden*2, K/V from cache → tmp ──
            self.bind_pipe(cmd, PipelineType::GqaAttention);
            pc.input_offset = tmp + M as u64 * hidden as u64 * 2; // Q start
            pc.weights_offset = cache_layer;                       // K/V cache
            pc.output_offset = tmp;
            pc.M = M; pc.K = s.seq_len + M; // total tokens in cache
            self.push(cmd, &pc);
            unsafe { dev.cmd_dispatch(cmd, M, 16, 1); }
            self.barrier(cmd);

            // ── Output GEMM: tmp(attn_out) × attn_output → out_off ──
            self.bind_pipe(cmd, PipelineType::AttnOutput);
            pc.input_offset = tmp;
            pc.weights_offset = self.weights.attn_output[layer as usize];
            pc.output_offset = out_off;
            pc.M = M; pc.N = hidden; pc.K = hidden;
            self.push(cmd, &pc);
            unsafe { dev.cmd_dispatch(cmd, self.div64(M), self.div8(hidden), 1); }
        } else {
            // ── DeltaNet QKV: tmp(norm) → tmp + hidden*2 ──
            self.bind_pipe(cmd, PipelineType::DeltaNetQkv);
            pc.input_offset = tmp;
            pc.weights_offset = self.weights.attn_q[layer as usize];
            pc.output_offset = tmp + M as u64 * hidden as u64 * 2;
            pc.M = M; pc.N = 4128; pc.K = hidden;
            self.push(cmd, &pc);
            unsafe { dev.cmd_dispatch(cmd, self.div64(M), self.div8(4128), 1); }
            self.barrier(cmd);

            // ── DeltaNet Step ──
            self.bind_pipe(cmd, PipelineType::DeltaNetStep);
            let state_off = self.layout.deltanet_state_base
                + layer as u64 * 16 * 32 * 128 * 128 * 4;
            pc.input_offset = tmp + M as u64 * hidden as u64 * 2;
            pc.weights_offset = state_off;
            pc.output_offset = state_off;
            pc.num_qk_heads = 16; pc.num_v_heads = 32; pc.layer_idx = layer;
            self.push(cmd, &pc);
            unsafe { dev.cmd_dispatch(cmd, 16, 32, 1); }
            self.barrier(cmd);

            // ── DeltaNet Output ──
            self.bind_pipe(cmd, PipelineType::DeltaNetOutput);
            pc.input_offset = tmp + M as u64 * hidden as u64 * 2;
            pc.weights_offset = self.weights.attn_output[layer as usize];
            pc.output_offset = out_off;
            pc.M = M; pc.N = hidden; pc.K = 4096;
            self.push(cmd, &pc);
        }
        self.barrier(cmd);

        // ── (M1) Residual Add: hi + out_off → ho ──
        self.bind_pipe(cmd, PipelineType::ResidualAdd);
        pc.input_offset = hi;       // original input (residual)
        pc.weights_offset = out_off; // GEMM output
        pc.output_offset = ho;      // write to output buffer (not hi!)
        pc.M = M; pc.N = hidden;
        self.push(cmd, &pc);
        unsafe { dev.cmd_dispatch(cmd, M, self.div256(hidden), 1); }
        self.barrier(cmd);

        // ── Router: ho × router_weights → routing_logits ──
        self.bind_pipe(cmd, PipelineType::Router);
        pc.input_offset = ho; // read from the NEW hidden state (after residual)
        pc.weights_offset = self.weights.ffn_gate[layer as usize];
        pc.output_offset = self.layout.routing_logits_base;
        pc.M = M; pc.N = self.weights.num_experts; pc.K = hidden;
        self.push(cmd, &pc);
        unsafe { dev.cmd_dispatch(cmd, self.div64(M), self.div8(self.weights.num_experts), 1); }

        unsafe { dev.end_command_buffer(cmd).unwrap(); }
        self.submit(cmd)
    }

    // ── Router CPU read (C3: reads from just-written routing_logits) ──

    fn record_and_submit_router(&self, _layer: u32, M: u32) -> Result<(), String> {
        // Router GEMM was already dispatched at the end of record_layer.
        // This hook is for future CPU-side sync if needed.
        Ok(())
    }

    fn do_route(&self, M: u32) -> Vec<router::RoutingOutput> {
        let base = self.layout.routing_logits_base as usize;
        let n_exp = self.weights.num_experts as usize;
        if base == 0 || M == 0 { return vec![]; }

        // (M4) Router writes f16, but CPU needs f32. Convert inline.
        let raw = unsafe {
            std::slice::from_raw_parts(
                self.arena_base.add(base) as *const u16, M as usize * n_exp)
        };
        let mut logits = vec![0.0f32; M as usize * n_exp];
        for (i, &f16_bits) in raw.iter().enumerate() {
            // f16 → f32 (same conversion as common.glsl)
            let sign = (f16_bits & 0x8000) as u32;
            let exp = ((f16_bits & 0x7C00) >> 10) as u32;
            let mant = (f16_bits & 0x03FF) as u32;
            let f32_bits = if exp == 0 {
                if mant == 0 { 0 } else { sign }
            } else if exp == 31 {
                sign | 0x7F800000 | (mant << 13)
            } else {
                sign | ((exp + 112) << 23) | (mant << 13)
            };
            logits[i] = f32::from_bits(f32_bits);
        }
        router::route_cpu(&logits, self.weights.num_experts, M)
    }

    // ── MoE recording (C2: adds bind_descriptor_sets) ──

    fn record_and_submit_moe(&self, layer: u32, routing: &[router::RoutingOutput], M: u32)
        -> Result<(), String> {
        let cmd = self.cmd_moe;
        let dev = &self.device.device;
        unsafe {
            dev.reset_fences(&[self.fence]).unwrap();
            dev.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        }
        let begin = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe { dev.begin_command_buffer(cmd, &begin).unwrap(); }

        let is_prefill = routing.len() > 1;
        let hi = self.hi(true);
        let hidden = self.weights.hidden_size;
        let inter = self.weights.intermediate_size;
        let scratch = self.scratch();

        if is_prefill {
            // ── Prefill: batch tokens by expert (H7/H8) ──
            let (sorted_tokens, sorted_weights, ranges) =
                router::build_expert_batches(routing, self.weights.num_experts);
            let total_slots = sorted_tokens.len() as u32;
            let tok_base = scratch;
            let wgt_base = scratch + 256 * 1024;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    sorted_tokens.as_ptr(),
                    self.arena_base.add(tok_base as usize) as *mut u16,
                    total_slots as usize);
                std::ptr::copy_nonoverlapping(
                    sorted_weights.as_ptr(),
                    self.arena_base.add(wgt_base as usize) as *mut f32,
                    total_slots as usize);
            }

            let mut pc = PushConstants::default();
            macro_rules! batch_exp {
                ($w1:expr, $w3:expr, $w2:expr, $e:expr) => {{
                    let cnt = (ranges[$e + 1] - ranges[$e]) as u32;
                    if cnt == 0 {  } else {
                    let to = tok_base + ranges[$e] as u64 * 2;
                    let wo = wgt_base + ranges[$e] as u64 * 4;
                    self.bind_pipe(cmd, PipelineType::W1W3Fused);
                    pc.input_offset = hi; pc.weights_offset = $w1; pc.routing_weights_off = $w3;
                    pc.output_offset = scratch + cnt as u64 * inter as u64 * 2;
                    pc.M = cnt; pc.N = inter * 2; pc.K = hidden; pc.token_ids_off = to;
                    self.push(cmd, &pc);
                    unsafe { dev.cmd_dispatch(cmd, self.div64(cnt), self.div8(inter * 2), 1); }
                    self.barrier(cmd);
                    self.bind_pipe(cmd, PipelineType::SiluMult);
                    pc.input_offset = scratch + cnt as u64 * inter as u64 * 2;
                    pc.M = cnt; pc.N = inter; pc.token_ids_off = 0;
                    self.push(cmd, &pc);
                    unsafe { dev.cmd_dispatch(cmd, cnt, self.div256(inter), 1); }
                    self.barrier(cmd);
                    self.bind_pipe(cmd, PipelineType::W2Scatter);
                    pc.input_offset = scratch + cnt as u64 * inter as u64 * 2;
                    pc.weights_offset = $w2; pc.output_offset = hi;
                    pc.M = cnt; pc.N = hidden; pc.K = inter;
                    pc.scale_factor = 1.0 / 9.0;
                    pc.token_ids_off = to; pc.routing_weights_off = wo;
                    self.push(cmd, &pc);
                    unsafe { dev.cmd_dispatch(cmd, self.div64(cnt), self.div8(hidden), 1); }
                    self.barrier(cmd);
                } } };
            }
            let sid = routing[0].shared_id as usize;
            batch_exp!(self.weights.shared_w1[layer as usize], self.weights.shared_w3[layer as usize], self.weights.shared_w2[layer as usize], sid);
            for e in 0..self.weights.num_experts as usize {
                let base = layer as usize * 256 + e;
                batch_exp!(self.weights.expert_w1[base], self.weights.expert_w3[base], self.weights.expert_w2[base], e);
            }
        } else {
            // ── Generation: sequential per expert (M=1) ──
            let mut pc = PushConstants::default();
            macro_rules! gen_exp {
                ($w1:expr, $w3:expr, $w2:expr, $wt:expr) => {{
                    self.bind_pipe(cmd, PipelineType::W1W3Fused);
                    pc.input_offset = hi; pc.weights_offset = $w1; pc.routing_weights_off = $w3;
                    pc.output_offset = scratch + M as u64 * inter as u64 * 2;
                    pc.M = M; pc.N = inter * 2; pc.K = hidden; pc.token_ids_off = 0;
                    self.push(cmd, &pc);
                    unsafe { dev.cmd_dispatch(cmd, self.div64(M), self.div8(inter * 2), 1); }
                    self.barrier(cmd);
                    self.bind_pipe(cmd, PipelineType::SiluMult);
                    pc.input_offset = scratch + M as u64 * inter as u64 * 2;
                    pc.M = M; pc.N = inter;
                    self.push(cmd, &pc);
                    unsafe { dev.cmd_dispatch(cmd, M, self.div256(inter), 1); }
                    self.barrier(cmd);
                    self.bind_pipe(cmd, PipelineType::W2Scatter);
                    pc.input_offset = scratch + M as u64 * inter as u64 * 2;
                    pc.weights_offset = $w2; pc.output_offset = hi;
                    pc.M = M; pc.N = hidden; pc.K = inter;
                    pc.scale_factor = $wt / 9.0; pc.token_ids_off = 0;
                    self.push(cmd, &pc);
                    unsafe { dev.cmd_dispatch(cmd, self.div64(M), self.div8(hidden), 1); }
                    self.barrier(cmd);
                }};
            }
            gen_exp!(self.weights.shared_w1[layer as usize], self.weights.shared_w3[layer as usize], self.weights.shared_w2[layer as usize], 1.0);
            if let Some(r) = routing.first() {
                for i in 0..8 {
                    let base = layer as usize * 256 + r.routed[i] as usize;
                    gen_exp!(self.weights.expert_w1[base], self.weights.expert_w3[base], self.weights.expert_w2[base], r.weights[i]);
                }
            }
        }
        unsafe { dev.end_command_buffer(cmd).unwrap(); }
        self.submit(cmd)
    }
}

impl Drop for InferenceEngine {
    fn drop(&mut self) {
        unsafe { self.device.device.device_wait_idle().ok(); }
    }
}
