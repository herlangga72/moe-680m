---
type: concept
title: Process Flows
date: 2026-07-25
bundle: moe-680m-docs
concepts: [architecture, inference-flow]
---

# Process Flows

This section documents the end-to-end business processes of the inference engine.

## States

### InferenceState

```
struct InferenceState {
    position: u32,       // current token position in sequence
    seq_len: u32,        // total tokens processed (KV cache filled)
    hidden_ping: bool,   // ping-pong buffer selector for hidden states
}
```

Tracks autoregressive generation state across calls. `seq_len` grows with each generated token. `hidden_ping` alternates per layer to select input/output buffers.

### SamplingContext

```
struct SamplingContext {
    rng_state: u64,            // xorshift PRNG state
    past_tokens: Vec<u32>,     // recently generated tokens
    token_frequencies: HashMap<u32, u32>,  // frequency penalty tracking
}
```

## Flows

### 1. Startup / Initialization

```
[main.rs] → parse CLI args → [tui::run()] or direct
    → [run_inference()]
        → mmap GGUF
        → GgufReader::parse()
        → Tokenizer::from_gguf_meta()
        → DeviceContext::init()    [Vulkan init + UMA find]
        → ArenaLayout::compute()
        → device.allocate_uma(layout.total_size)
        → TensorRegistry::from_tensors()
        → load_weights_from_tensors()
        → create_pipelines()
        → bind_arena_descriptor()
        → InferenceEngine::new()
    → generation or server mode
```

### 2. Prefill (Prompt Encoding)

```
[M > 1 tokens] → generate_mtp(tokens, state)
    → prefill(tokens, state)
        → for each token: embed_token(id, position)
        → for each of 40 layers (unfused, M > 1):
            → record_and_submit_layer(layer, is_gqa, M, state)
                [individual dispatches: RMSNorm → QKV → RoPE/KVWrite/Attention → Output → Residual → Router]
            → do_route(M) [CPU softmax + top-8 + prefix-sum]
            → record_and_submit_moe(layer, routing, M, ping)
                [W1W3Fused + W2Scatter per expert batch]
    → state.seq_len = M
```

### 3. Generation (Single Token)

```
[M=1] → generate_mtp(tokens, state)
    → forward_single(state)
        → for each of 40 layers (fused, M=1):
            → if GQA:
                FusedRmsNormQkvRope → KVWrite → Attention → AttnResidual
            → if DeltaNet:
                FusedRmsNormDnQkv → DeltaNetStep → DnOutputResidual
            → Router (GPU topk when M=1)
        → MoE dispatch (sequential per expert)
        → LM Head GEMM → f16→f32 SIMD → sampling
    → read_mtp_drafts() [f16→f32 SIMD + argmax]
    → embed accepted tokens (main + drafts)
    → verification pass (all layers, M=n_accepted, unfused)
    → extend KV cache
```

### 4. Multi-Turn Chat

```
[chat_loop]
    → initial generation (same as CLI)
    → loop:
        → print "> " prompt
        → read user input (stdin)
        → if /exit or empty → break
        → append user message to conversation text
        → encode full conversation + BOS
        → generate_mtp() → incremental prefill + generation
        → stream response (token-by-token)
        → append response to conversation text
        → loop
```

### 5. HTTP Server (feature-gated)

```
[server::serve]
    → TcpListener on :8080
    → for each connection:
        → parse HTTP request
        → if POST /v1/messages:
            → JSON deserialize (Messages API format)
            → build prompt from messages array
            → tokenize
            → lock Mutex<InferenceEngine>
            → generate tokens
            → JSON or SSE response
```

## Data Flow Per Layer

```ascii
           ┌─ hidden_ping/pong (input)
           │
           ▼
      ┌──────────┐
      │ RMS Norm │  (or fused into rms_norm_qkv_rope / rms_norm_dn_qkv)
      └────┬─────┘
           │ f16 hidden, 2048 elements
           ▼
      ┌──────────┐
      │ QKV Proj │  (GQA: 3 GEMMs → Q4096 + K512 + V512)
      │          │  (DN: 1 GEMM → 4128 outputs)
      └────┬─────┘
           │ f16 QKV buffer
           ▼
      ┌──────────┐
      │ RoPE     │  (GQA only, first 64 dims of each Q/K head)
      └────┬─────┘
           ▼
      ┌──────────┐
      │ KV Cache │  (GQA: Q4_0 blocks at position seq_len)
      │ Write    │
      └────┬─────┘
           ▼
      ┌──────────┐
      │ Attention│  (GQA: online softmax, Q×K reduction)
      │ /Step    │  (DN: S = g·S + (1-g)·outer(k,v))
      └────┬─────┘
           │ f32/f16 attention output
           ▼
      ┌──────────┐
      │ Output   │  (GEMM: attention → hidden)
      │ Proj     │
      └────┬─────┘
           │ f16 output + residual (hi)
           ▼
      ┌──────────┐
      │ Router   │  (GEMM: hidden × router_weights → 256 logits)
      └────┬─────┘
           │ f16 routing logits
           ▼
      ┌──────────┐
      │ MoE      │  (W1W3Fused → W2Scatter per expert)
      │ FFN      │
      └────┬─────┘
           │ f16 hidden_out (= next layer's input)
           ▼
      ┌──────────┐
      │ (next layer)
      ▼
```

## Business Process: Fused vs Unfused Dispatch Selection

```ascii
record_and_submit_layer(layer, is_gqa, M, state):
    if M == 1 AND fused pipeline available:
        if is_gqa:
            use FusedRmsNormQkvRope  (1 dispatch vs 5)
            → KVWrite (1 dispatch)
            → Attention (1 dispatch)
            → AttnResidual (1 dispatch vs 2)
        else:
            use FusedRmsNormDnQkv (1 dispatch vs 2)
            → DeltaNetStep (1 dispatch, recurrent)
            → DnOutputResidual (1 dispatch vs 2)
    else:
        if is_gqa:
            RMSNorm → QGEMM → KGEMM → VGEMM → RoPE → KVWrite → Attention → AttnOutput
        else:
            RMSNorm → DnQkv → DnStep → DnOutput
        → ResidualAdd
    
    → Router (GEMM)
    → GPU TopK (if M==1)
```
