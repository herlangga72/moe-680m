use std::collections::HashMap;

pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub max_tokens: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 0, max_tokens: 4096 }
    }
}

pub struct SamplingContext {
    pub rng_state: u64,
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

impl Default for SamplingContext {
    fn default() -> Self {
        Self { rng_state: 0, past_tokens: Vec::new(), token_frequencies: HashMap::new() }
    }
}

/// Sample from logits given parameters.
pub fn sample(logits: &mut [f32], params: &SamplingParams, ctx: &mut SamplingContext) -> u32 {
    // Greedy fast path
    if params.temperature == 0.0 {
        return argmax(logits);
    }

    // Temperature
    let inv_temp = 1.0 / params.temperature;
    for l in logits.iter_mut() { *l *= inv_temp; }

    // Top-k
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

    // Sample (xorshift)
    let r = xorshift_f32(&mut ctx.rng_state);
    let mut cum = 0.0;
    for (i, &p) in logits.iter().enumerate() {
        cum += p;
        if r < cum { return i as u32; }
    }
    0
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn xorshift_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_add(1);
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    (x >> 11) as f32 * (1.0 / 9007199254740992.0)
}
