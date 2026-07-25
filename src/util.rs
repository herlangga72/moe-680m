// Shared f16 conversion helpers

pub const VOCAB_SIZE: u32 = 248320;

pub fn read_u16(base: *mut u8, off: usize) -> u16 {
    unsafe { (*(base.add(off)) as u16) | ((*(base.add(off + 1)) as u16) << 8) }
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    // Branchless: denormals flush to ~zero, inf/nan map to large finite.
    // Correct for inference — shaders never produce edge-case f16 values.
    let fb = bits as u32;
    f32::from_bits(
        ((fb & 0x8000) << 16)
        | (((fb >> 10) & 0x1F).wrapping_add(112) << 23)
        | ((fb & 0x3FF) << 13)
    )
}

pub fn f32_to_f16_bits(v: f32) -> u16 {
    // Branchless: clamp exponent, overflow maps to inf via arithmetic mask.
    let fb = v.to_bits();
    let s = ((fb >> 16) & 0x8000) as u16;
    // e32.clamp(-15, 16) → cmov on x86_64, not a branch
    let e = (((fb >> 23) & 0xFF) as i32 - 127).clamp(-15, 16);
    let m = ((fb >> 13) & 0x3FF) as u16;
    let f16 = s | (((e + 15) as u16) << 10) | m;
    // Overflow mask: replace with inf when f32 exponent >= 16
    let overflow = ((((fb >> 23) & 0xFF) as i32 - 127) >= 16) as u16;
    let mask = overflow.wrapping_neg(); // 0xFFFF if overflow, 0x0000 otherwise
    (f16 & !mask) | (mask & (s | (31 << 10)))
}
