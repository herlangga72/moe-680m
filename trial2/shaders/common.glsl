// ── Push constant layout ──

layout(push_constant, std430) uniform PC {
    uint rows;
    uint cols;
    uint stride;
    uint pad0;
    float param0;
    float param1;
    float param2;
    float param3;
    uint opt0;
    uint opt1;
    uint opt2;
    uint opt3;
} pc;

// ── FP16 helpers (RDNA2 native via VK_KHR_shader_float16_int8) ──

float16_t load_f16(uint addr) {
    return uint16BitsToFloat16(data16[addr >> 1]);
}

void store_f16(uint addr, float16_t val) {
    data16[addr >> 1] = float16BitsToUint16(val);
}

// ── iQ4_XS dequant (fused into weight-reading shaders) ──
// Block: 256 elements → 162 bytes
//   super-block: d (FP16, 2B)
//   8 sub-blocks of 32: m (FP16, 2B) + scales (2B packed: 8 × 2-bit)
//   values: 128B (256 × 4-bit nibbles)

float dequant_iq4_xs(uint blk_start, uint elem_idx) {
    uint blk = elem_idx / 256u;
    uint sub = (elem_idx % 256u) / 32u;
    uint elem_in_sub = elem_idx % 32u;

    // Block base in uint16 units
    uint base = 1296u * blk; // 162 bytes * 8 = 1296 uint16 units

    // Super-block scale d (first 2 bytes → 1 uint16)
    float d = float(data16[base]);
    base += 1u;

    // Skip to sub-block: m (1 uint16) + scales (1 uint16) per sub-block
    base += sub * 2u;

    // Sub-block min m
    float m = float(data16[base]);
    base += 1u;

    // Packed scales: 8 × 2-bit in 2 bytes
    uint scale_packed = data16[base];
    uint scale_bits = (scale_packed >> (sub * 2u)) & 0x3u;
    // Scale mapping for iQ4_XS: 2-bit → float
    // iQ4_XS 2-bit scale → float lookup (6 possible values encoded in 2 bits)
    float scale_table[4] = float[4](-0.5f, 0.0f, 0.5f, 1.0f);
    float scale = scale_table[scale_bits];

    // Nibble value (skip super-block header: 2 + 8*(2+2) = 34 bytes → 17 uint16)
    base = 1296u * blk + 17u;
    uint byte_idx = elem_in_sub / 2u;
    uint nibble = (data16[base + byte_idx / 2u] >> ((byte_idx & 1u) * 4u)) & 0xFu;

    return d * scale * float(nibble) + m;
}

// ── Q4_0 K dequant (for attention) ──
// Block: 32 elements → 18 bytes
//   d (FP16, 2B) + values (16 × 4-bit = 16B)

float16_t dequant_q4_0(uint blk_start, uint elem_idx) {
    uint blk = elem_idx / 32u;
    uint elem_in_blk = elem_idx % 32u;
    uint base = 9u * blk; // 18 bytes = 9 uint16

    float16_t d = float16_t(data16[base]);
    uint byte_idx = elem_in_blk / 2u;
    uint nibble = (data16[base + 1u + byte_idx / 2u] >> ((byte_idx & 1u) * 4u)) & 0xFu;

    return d * float16_t(nibble) - d * float16_t(8.0hf);
}

// ── Q8_0 dequant (for QKV, output projections) ──
// Block: 32 elements → 34 bytes
//   d (FP16 scale, 2B) + 32 × int8 values

float dequant_q8_0(uint blk_start, uint elem_idx) {
    uint blk = elem_idx / 32u;
    uint elem_in_blk = elem_idx % 32u;
    uint base = 17u * blk; // 34 bytes = 17 uint16 units

    // Scale d from first uint16
    float d = float(uint16BitsToFloat16(data16[base]));
    // int8 value at offset 2 + elem_in_blk (in uint8 units)
    uint byte_off = blk * 34u + 2u + elem_in_blk;
    float q = float(int(data8[byte_off]));
    return d * q;
}

// ── IQ3_S dequant (for FFN expert weights) ──
// 256-element blocks, ~3.4 bits/elem

float dequant_iq3_s(uint blk_start, uint elem_idx) {
    // ponytail: approximate — IQ3_S is complex, use Q4_0 fallback for now
    uint blk = elem_idx / 32u;
    uint base = 9u * blk;
    float16_t d = float16_t(data16[base]);
    uint byte_idx = elem_idx % 32u / 2u;
    uint nibble = (data16[base + 1u + byte_idx / 2u] >> ((byte_idx & 1u) * 4u)) & 0xFu;
    return float(d) * float(nibble) - float(d) * 8.0;
}

// ── Generic dequant dispatcher (uses pc.opt0) ──
// 0=FP32 (read float directly), 1=Q8_0, 2=IQ4_XS, 3=IQ3_S

float dequant(uint blk_start, uint elem_idx) {
    switch (pc.opt0) {
        case 1u: return dequant_q8_0(blk_start, elem_idx);
        case 2u: return dequant_iq4_xs(blk_start, elem_idx);
        case 3u: return dequant_iq3_s(blk_start, elem_idx);
        default: return data[blk_start + elem_idx]; // FP32 passthrough
    }
}

// ── Int8 V unpack (for attention) ──

float16_t unpack_int8_v(uint addr) {
    uint word = data8[addr >> 2];
    uint shift = (addr & 3u) * 8u;
    int val = int((word >> shift) & 0xFFu);
    if ((val & 0x80) != 0) val |= ~0xFF;
    return float16_t(val);
}

// ── Subgroup reductions (Wave64 native) ──

float subgroup_sum(float v) {
    return subgroupAdd(v);
}

float subgroup_max(float v) {
    return subgroupMax(v);
}

// ── RMSNorm helper ──

float rsqrt_fast(float x) {
    return inversesqrt(x);
}
