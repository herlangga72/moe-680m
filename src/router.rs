// Job 9: CPU-side MoE routing
// 256-way softmax + top-8 selection + prefix-sum bucket fill.

pub struct RoutingOutput {
    pub routed: [u16; 8],     // top-8 expert indices
    pub weights: [f32; 8],    // softmax weights for top-8
    pub shared_id: u16,       // shared expert index (always active)
}

/// Read routing logits from coherent GPU memory, compute softmax + top-8.
/// `logits_base` is the CPU-accessible pointer to the [num_tokens × 256] f32 array.
pub fn route_cpu(logits: &[f32], num_experts: u32, num_tokens: u32) -> Vec<RoutingOutput> {
    let mut results = Vec::with_capacity(num_tokens as usize);
    for t in 0..num_tokens as usize {
        let base = t * num_experts as usize;
        let token_logits = &logits[base..base + num_experts as usize];
        results.push(route_single(token_logits));
    }
    results
}

pub(crate) fn route_single(logits: &[f32]) -> RoutingOutput {
    // Softmax
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = [0.0f32; 256];
    let mut sum = 0.0;
    for i in 0..logits.len().min(256) {
        probs[i] = (logits[i] - max).exp();
        sum += probs[i];
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for p in &mut probs { *p *= inv; }
    }

    // Top-8 via partial sort: O(n) instead of O(n log n)
    let n = logits.len().min(256);
    let mut indices = [0usize; 256];
    for i in 0..n { indices[i] = i; }
    indices[..n].select_nth_unstable_by(7, |&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut routed = [0u16; 8];
    let mut weights = [0.0f32; 8];
    for i in 0..8 {
        routed[i] = indices[i] as u16;
        weights[i] = probs[indices[i]];
    }

    RoutingOutput { routed, weights, shared_id: 0 }
}

/// Prefill routing: build per-expert token assignment table.
/// Returns (sorted_tokens, sorted_weights, expert_ranges[257]).
pub fn build_expert_batches(
    routing: &[RoutingOutput],
    num_experts: u32,
) -> (Vec<u16>, Vec<f32>, Vec<u32>) {
    let total_slots = routing.len() as u32 * 9; // 8 routed + 1 shared per token
    let mut counts = vec![0u32; num_experts as usize];

    // Count per expert
    for r in routing {
        for &e in &r.routed {
            counts[e as usize] += 1;
        }
        counts[r.shared_id as usize] += 1;
    }

    // Prefix sum
    let mut ranges = vec![0u32; num_experts as usize + 1];
    for i in 0..num_experts as usize {
        ranges[i + 1] = ranges[i] + counts[i];
    }

    // Fill sorted arrays
    let mut sorted_tokens = vec![0u16; total_slots as usize];
    let mut sorted_weights = vec![0.0f32; total_slots as usize];
    let mut cursors = ranges.clone();

    for (t, r) in routing.iter().enumerate() {
        for i in 0..8 {
            let e = r.routed[i] as usize;
            let pos = cursors[e] as usize;
            cursors[e] += 1;
            sorted_tokens[pos] = t as u16;
            sorted_weights[pos] = r.weights[i];
        }
        let e = r.shared_id as usize;
        let pos = cursors[e] as usize;
        cursors[e] += 1;
        sorted_tokens[pos] = t as u16;
        sorted_weights[pos] = 1.0;
    }

    (sorted_tokens, sorted_weights, ranges)
}

