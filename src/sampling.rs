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
    // Greedy fast path (LLVM vectorizes argmax with maxps)
    if params.temperature == 0.0 {
        return crate::util::argmax(logits);
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

    // Softmax: exp
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for l in logits.iter_mut() {
        *l = (*l - max).exp();
        sum += *l;
    }
    // Flush zero-distribution (all logits were -inf or similar)
    if sum <= 0.0 { return 0; }

    // Normalize and build CDF in-place (overwrites probs with cumulative)
    let inv = 1.0 / sum;
    let mut cum = 0.0;
    for l in logits.iter_mut() {
        cum += *l * inv;
        *l = cum;
    }

    // Branchless binary search on CDF
    // Comparison compiles to ucomiss + setbe (no branch), mask arithmetic for lo/hi
    let r = xorshift_f32(&mut ctx.rng_state);
    let mut lo = 0usize;
    let mut hi = logits.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let go_right = (logits[mid] <= r) as usize;    // setcc, no branch
        let mask = go_right.wrapping_neg();             // 0 or !0
        lo = ((mid + 1) & mask) | (lo & !mask);
        hi = (mid & !mask) | (hi & mask);
    }
    // Guard: if r > cdf[last] (FP epsilon), cap at last valid index
    (lo as u32).min(logits.len() as u32 - 1)
}

// argmax moved to crate::util::argmax (SIMD-friendly manual loop)

fn xorshift_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_add(1);
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    (x >> 11) as f32 * (1.0 / 9007199254740992.0)
}
