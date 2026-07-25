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

uint read_u32(uint64_t off) {
    return data[uint(off >> 2)];  // dword-aligned read from byte offset
}

// ── f16 ↔ f32 conversion (native RDNA2 instructions) ──
// unpackHalf2x16 → v_cvt_f32_f16 (1-2 instr), packHalf2x16 → v_cvt_pkrtz_f32_f16 (1 instr).

float f16_to_f32(uint bits) {
    return unpackHalf2x16(bits).x;  // lower 16 bits → f32, native speed
}

uint f32_to_f16(float v) {
    return packHalf2x16(vec2(v, 0.0));  // f32 → f16 in lower 16 bits, HW rounding
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

    // Both f16 scales in one dword: unpackHalf2x16 extracts both via single v_cvt_f32_f16 pair
    vec2 d_scale = unpackHalf2x16(read_u32(bo));
    float d = (block_pos < 16u) ? d_scale.x : d_scale.y;

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
