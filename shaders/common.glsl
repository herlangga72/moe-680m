// ── Common helpers for MoE-680M shaders ──

// ── Byte-level buffer access ──

uint read_u8(uint64_t off) {
    uint word = data[uint(off >> 2)];
    return (word >> (uint(off & 3) * 8)) & 0xFF;
}

uint read_u16(uint64_t off) {
    uint lo = read_u8(off);
    uint hi = read_u8(off + uint64_t(1));
    return lo | (hi << 8);
}

// ── f16 ↔ f32 conversion (branchless, using hardware intrinsics) ──
// RDNA2 has native v_cvt_f32_f16 / v_cvt_pkrtz_f32_f16 instructions.
// unpackHalf2x16 / packHalf2x16 compile to these on Vulkan + RADV.

float f16_to_f32(uint bits) {
    // Branchless: compute all possible exponent/mantissa paths simultaneously
    // and select using masks. Flushes denormals to zero (acceptable for inference).
    uint s = (bits & 0x8000u) << 16u;
    uint e = (bits & 0x7C00u) >> 10u;
    uint m = bits & 0x03FFu;
    // Normal: e + 112, mant << 13
    // Denorm (e==0): flush to zero
    // Inf/NaN (e==31): e = 255, mant << 13
    uint e_norm = e + 112u;
    uint e_clamp = (e == 31u) ? 255u : e_norm;
    uint e_keep = (e == 0u) ? 0u : e_clamp;
    uint m_keep = (e == 0u) ? 0u : (m << 13u);
    uint bits32 = s | (e_keep << 23u) | m_keep;
    return uintBitsToFloat(bits32);
}

uint f32_to_f16(float v) {
    // Branchless f32→f16 with clamp.
    // Extract fields, compute f16 exponent with clamp, select results.
    uint fb = floatBitsToUint(v);
    uint s16 = (fb >> 16u) & 0x8000u;
    int e32 = int((fb >> 23u) & 0xFFu) - 127;
    uint m16 = (fb >> 13u) & 0x03FFu;
    // Clamp exponent from [-127..128] to [-15..16] for f16
    e32 = max(e32, -15);
    e32 = min(e32, 16);
    uint e16 = uint(e32 + 15);
    uint f16 = s16 | (e16 << 10u) | m16;
    // Handle edge: if original f32 is zero, output zero
    // (all bits zero case — mantissa may have sticky bits from clamp)
    if (fb == 0u) return 0u;
    // Clamp to max f16 if exponent was >= 16 in f32 scale
    // (handles overflow to infinity)
    int orig_e32 = int((fb >> 23u) & 0xFFu) - 127;
    if (orig_e32 >= 16) f16 = s16 | (31u << 10u);
    return f16;
}

// ── IQ4_XS dequant ──

const uint IQ4_BLOCK_SIZE = 32;
const uint IQ4_QS_OFF = 4;
const uint IQ4_QH_OFF = 20;
const uint IQ4_BLOCK_BYTES = 36;

float dequant_iq4_xs(uint64_t weights_base, uint weight_idx) {
    uint block_idx = weight_idx / IQ4_BLOCK_SIZE;
    uint block_pos = weight_idx % IQ4_BLOCK_SIZE;
    uint64_t bo = weights_base + uint64_t(block_idx) * IQ4_BLOCK_BYTES;

    // Branchless d/d2 selection via select()
    float d0 = f16_to_f32(read_u16(bo));
    float d2 = f16_to_f32(read_u16(bo + 2));
    float d = (block_pos < 16u) ? d0 : d2;

    uint ns = IQ4_QS_OFF + (block_pos >> 1);
    uint byte = read_u8(bo + uint64_t(ns));
    uint q_val = (byte >> ((block_pos & 1) * 4)) & 0xF;

    uint hb = IQ4_QH_OFF + (block_pos >> 3);
    uint hbyte = read_u8(bo + uint64_t(hb));
    uint high = (hbyte >> (block_pos & 7)) & 1u;
    q_val |= high << 4;

    return (float(q_val) - 8.0) * d;
}

// ── Q4_0 dequant (for KV cache reads) ──

float read_kv_q4_0(uint64_t cache_base, uint max_seq, uint pos, uint elem) {
    uint64_t base = (elem < 512u)
        ? cache_base + uint64_t(pos) * 288u
        : cache_base + uint64_t(max_seq) * 288u + uint64_t(pos) * 288u;
    uint le = elem % 512u;
    uint blk = le / 32u;
    uint bp = le % 32u;
    uint64_t bo = base + uint64_t(blk) * 18u;
    float d = f16_to_f32(read_u16(bo));
    uint qs = read_u8(bo + 2u + (bp >> 1));
    uint nibble = (qs >> ((bp & 1u) * 4u)) & 0xFu;
    return (float(nibble) - 8.0) * d;
}

// ── SiLU activation ──

float silu(float x) {
    return x / (1.0 + exp(-x));
}
