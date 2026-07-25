// Shared f16 conversion helpers

pub const VOCAB_SIZE: u32 = 248320;

pub fn read_u16(base: *mut u8, off: usize) -> u16 {
    unsafe { (*(base.add(off)) as u16) | ((*(base.add(off + 1)) as u16) << 8) }
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let s = (bits as u32 & 0x8000) << 16;
    let e = (bits >> 10) as u32 & 0x1F;
    let m = (bits & 0x03FF) as u32;
    if e == 0 { return 0.0; }
    if e == 31 { return if m == 0 { f32::INFINITY } else { f32::NAN }; }
    let e_norm = e + 112;
    f32::from_bits(s | (e_norm << 23) | (m << 13))
}

pub fn f32_to_f16_bits(v: f32) -> u16 {
    let fb = v.to_bits();
    let s16 = ((fb >> 16) & 0x8000) as u16;
    let e32 = ((fb >> 23) & 0xFF) as i32 - 127;
    let m16 = ((fb >> 13) & 0x03FF) as u16;
    if fb == 0 { return 0; }
    let e32 = e32.clamp(-15, 16);
    let e16 = (e32 + 15) as u16;
    let mut f16 = s16 | (e16 << 10) | m16;
    let orig_e32 = ((fb >> 23) & 0xFF) as i32 - 127;
    if orig_e32 >= 16 { f16 = s16 | (31 << 10); }
    f16
}
