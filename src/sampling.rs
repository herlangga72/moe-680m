// Jobs 11-12: Sampling + stop conditions + multi-turn
// See plans/tier1-sampling.md for full design.

use std::collections::HashMap;

pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub max_tokens: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: 4096,
        }
    }
}

#[derive(Default)]
pub struct SamplingContext {
    pub past_tokens: Vec<u32>,
    pub token_frequencies: HashMap<u32, u32>,
}

impl SamplingContext {
    pub fn record(&mut self, token: u32) {
        self.past_tokens.push(token);
        *self.token_frequencies.entry(token).or_insert(0) += 1;
    }

    pub fn reset(&mut self) {
        self.past_tokens.clear();
        self.token_frequencies.clear();
    }
}

/// Sample from logits given parameters.
pub fn sample(logits: &mut [f32], params: &SamplingParams, ctx: &SamplingContext) -> u32 {
    // Greedy fast path
    if params.temperature == 0.0 {
        return argmax(logits);
    }

    // Temperature
    let inv_temp = 1.0 / params.temperature;
    for l in logits.iter_mut() { *l *= inv_temp; }

    // Repetition penalty
    if params.repetition_penalty != 1.0 {
        for &token in &ctx.past_tokens {
            let idx = token as usize;
            if idx < logits.len() {
                if logits[idx] < 0.0 { logits[idx] *= params.repetition_penalty; }
                else { logits[idx] /= params.repetition_penalty; }
            }
        }
    }

    // Frequency / presence penalty
    if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
        for (&token, &count) in &ctx.token_frequencies {
            let idx = token as usize;
            if idx < logits.len() {
                logits[idx] -= params.frequency_penalty * count as f32;
                logits[idx] -= params.presence_penalty;
            }
        }
    }

    // Top-k (simple approach: sort indices by value)
    if params.top_k > 0 {
        let k = (params.top_k as usize).min(logits.len());
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        for (pos, &idx) in indices.iter().enumerate() {
            if pos >= k { logits[idx] = f32::NEG_INFINITY; }
        }
    }

    // Softmax
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for l in logits.iter_mut() {
        *l = (*l - max).exp();
        sum += *l;
    }
    if sum > 0.0 { let inv = 1.0 / sum; for l in logits.iter_mut() { *l *= inv; } }

    // Top-p (nucleus)
    if params.top_p < 1.0 {
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let mut cum = 0.0;
        for &idx in &indices {
            cum += logits[idx];
            if cum > params.top_p { logits[idx] = 0.0; }
        }
        let sum2: f32 = logits.iter().sum();
        if sum2 > 0.0 { let inv = 1.0 / sum2; for l in logits.iter_mut() { *l *= inv; } }
    }

    // Sample (xorshift)
    let r = xorshift_f32();
    let mut cum = 0.0;
    for (i, &p) in logits.iter().enumerate() {
        cum += p;
        if r < cum { return i as u32; }
    }
    0
}

pub fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ── Stop conditions ──

// ── Simple xorshift RNG (no external dep) ──
static mut RNG_STATE: u64 = 0;

fn xorshift_f32() -> f32 {
    unsafe {
        RNG_STATE = RNG_STATE.wrapping_add(1);
        let mut x = RNG_STATE;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f32 * (1.0 / 9007199254740992.0)
    }
}

pub enum StopReason {
    NotStopped,
    MaxTokens,
    EosToken(u32),
    StopString(String),
}

pub struct StopConditions {
    pub max_tokens: u32,
    pub eos_token_id: u32,
    pub stop_strings: Vec<String>,
}

impl Default for StopConditions {
    fn default() -> Self {
        Self { max_tokens: 4096, eos_token_id: u32::MAX, stop_strings: vec![] }
    }
}

pub fn check_stops(
    token: u32,
    decoded: &str,
    tokens_generated: u32,
    conditions: &StopConditions,
) -> StopReason {
    if tokens_generated >= conditions.max_tokens {
        return StopReason::MaxTokens;
    }
    if token == conditions.eos_token_id {
        return StopReason::EosToken(token);
    }
    for s in &conditions.stop_strings {
        if decoded.contains(s.as_str()) {
            return StopReason::StopString(s.clone());
        }
    }
    StopReason::NotStopped
}

// ── Multi-turn session ──

pub struct ConversationSession {
    pub messages: Vec<(String, String)>, // (role, content)
    pub kv_cache_filled: u32,
    pub sampling_ctx: SamplingContext,
}

impl ConversationSession {
    pub fn new() -> Self {
        Self { messages: vec![], kv_cache_filled: 0, sampling_ctx: SamplingContext::default() }
    }

    pub fn add(&mut self, role: &str, content: &str) {
        self.messages.push((role.to_string(), content.to_string()));
    }

    pub fn reset(&mut self) {
        self.messages.clear();
        self.kv_cache_filled = 0;
        self.sampling_ctx.reset();
    }
}
