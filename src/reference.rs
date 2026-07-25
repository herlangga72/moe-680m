// ── CPU reference implementations for GPU validation ──
// Scalar, unoptimized. Only for cross-checking GPU output.

use crate::gguf::GgmlType;

/// IQ4_XS dequant: one weight at linear index `idx` from block starting at `data`.
pub fn ref_dequant_iq4_xs(data: &[u8], block_idx: usize, block_pos: usize) -> f32 {
    let bo = block_idx * 36; // BLOCK_BYTES
    // d (f16) at offset 0, d2 (f16) at offset 2
    let d = if block_pos < 16 {
        f16_to_f32(data[bo], data[bo+1])
    } else {
        f16_to_f32(data[bo+2], data[bo+3])
    };
    // qs nibble at offset 4
    let ns = 4 + (block_pos >> 1);
    let byte = data[ns];
    let q_val = (byte >> ((block_pos & 1) * 4)) & 0xF;
    // qh high bit at offset 20
    let hb = 20 + (block_pos >> 3);
    let high = (data[hb] >> (block_pos & 7)) & 1;
    let q_val = q_val | (high << 4);
    (q_val as f32 - 8.0) * d
}

/// Reference dequant GEMM: C[M,N] = A[M,K] × B[K,N] (IQ4_XS weights).
pub fn ref_dequant_gemm(a: &[f16], b_q4: &[u8], c: &mut [f32],
                         M: usize, N: usize, K: usize) {
    for m in 0..M {
        for n in 0..N {
            let mut sum = 0.0f32;
            for k in 0..K {
                let a_val = a[m * K + k] as f32;
                let b_val = ref_dequant_iq4_xs(b_q4, (k * N + n) / 32, (k * N + n) % 32);
                sum += a_val * b_val;
            }
            c[m * N + n] = sum;
        }
    }
}

/// Reference RMS norm: y = x / sqrt(mean(x^2) + eps).
pub fn ref_rms_norm(x: &[f32], y: &mut [f32], eps: f32) {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();
    for (i, &v) in x.iter().enumerate() {
        y[i] = v / rms;
    }
}

/// Reference single-head attention with online softmax.
pub fn ref_attention(q: &[f32], k: &[f32], v: &[f32], output: &mut [f32],
                     seq_len: usize, head_dim: usize) {
    let mut m = f32::NEG_INFINITY;
    let mut d = 0.0f32;
    let mut acc = vec![0.0f32; head_dim];

    for pos in 0..seq_len {
        let mut score = 0.0f32;
        for i in 0..head_dim {
            score += q[i] * k[pos * head_dim + i];
        }
        let new_m = m.max(score);
        let rescale = (m - new_m).exp();
        let exp_score = (score - new_m).exp();
        d = d * rescale + exp_score;
        for i in 0..head_dim {
            acc[i] = acc[i] * rescale + exp_score * v[pos * head_dim + i];
        }
        m = new_m;
    }

    for i in 0..head_dim {
        output[i] = if d > 0.0 { acc[i] / d } else { 0.0 };
    }
}

/// Reference DeltaNet step: S = gate*S + (1-gate)*k⊗v.
pub fn ref_deltanet_step(s: &mut [f32], k: &[f32], v: &[f32], q: &[f32],
                          gate: f32, output: &mut [f32], dim: usize) {
    let omg = 1.0 - gate;
    for i in 0..dim {
        for j in 0..dim {
            s[i * dim + j] = gate * s[i * dim + j] + omg * k[i] * v[j];
        }
    }
    // output = S × q
    for i in 0..dim {
        let mut sum = 0.0;
        for j in 0..dim {
            sum += s[i * dim + j] * q[j];
        }
        output[i] = sum;
    }
}

/// f16 bytes → f32 (software, matches GLSL branchless version).
fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = (hi as u16) << 8 | lo as u16;
    let sign = (bits as u32 & 0x8000) << 16;
    let mut exp = (bits >> 10) as u32 & 0x1F;
    let mant = (bits & 0x03FF) as u32;
    if exp == 0 {
        if mant == 0 { return 0.0; }
        // Denorm: flush to zero for simplicity
        return 0.0;
    }
    if exp == 31 {
        return if mant == 0 { f32::INFINITY } else { f32::NAN };
    }
    exp = exp + 112;
    let f32_bits = sign | (exp << 23) | (mant << 13);
    f32::from_bits(f32_bits)
}

// Alias for use by other modules
pub type f16 = half_f16;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct half_f16(pub u16);

impl From<half_f16> for f32 {
    fn from(h: half_f16) -> Self {
        f16_to_f32((h.0 & 0xFF) as u8, (h.0 >> 8) as u8)
    }
}
