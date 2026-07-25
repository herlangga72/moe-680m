// Shared f16 conversion helpers

pub const VOCAB_SIZE: u32 = 248320;

pub fn read_u16(base: *mut u8, off: usize) -> u16 {
    unsafe { (*(base.add(off)) as u16) | ((*(base.add(off + 1)) as u16) << 8) }
}

// ── Scalar f16↔f32 (branchless arithmetic) ──

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let fb = bits as u32;
    f32::from_bits(
        ((fb & 0x8000) << 16)
        | (((fb >> 10) & 0x1F).wrapping_add(112) << 23)
        | ((fb & 0x3FF) << 13)
    )
}

pub fn f32_to_f16_bits(v: f32) -> u16 {
    let fb = v.to_bits();
    let s = ((fb >> 16) & 0x8000) as u16;
    let e = (((fb >> 23) & 0xFF) as i32 - 127).clamp(-15, 16);
    let m = ((fb >> 13) & 0x3FF) as u16;
    let f16 = s | (((e + 15) as u16) << 10) | m;
    let overflow = ((((fb >> 23) & 0xFF) as i32 - 127) >= 16) as u16;
    let mask = overflow.wrapping_neg();
    (f16 & !mask) | (mask & (s | (31 << 10)))
}

// ── SIMD batch f16→f32 (F16C vcvtph2ps when available) ──

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f16_slice_to_f32_avx(src: &[u16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    let mut i = 0usize;
    while i + 8 <= n {
        let v = std::arch::x86_64::_mm_loadu_si128(src.as_ptr().add(i) as *const _);
        let f = std::arch::x86_64::_mm256_cvtph_ps(v);
        std::arch::x86_64::_mm256_storeu_ps(dst.as_mut_ptr().add(i), f);
        i += 8;
    }
    for j in i..n {
        dst[j] = f16_bits_to_f32(src[j]);
    }
}

pub fn f16_slice_to_f32(src: &[u16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 8
        && std::arch::is_x86_feature_detected!("avx")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        unsafe { f16_slice_to_f32_avx(src, dst); }
        return;
    }
    for i in 0..n {
        dst[i] = f16_bits_to_f32(src[i]);
    }
}

// ── Argmax: index of max element (LLVM auto-vectorizes with maxps) ──

pub fn argmax(slice: &[f32]) -> u32 {
    let mut best_i = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as u32;
        }
    }
    best_i
}

// ── SIMD batch f32→f16 (F16C vcvtps2ph when available) ──

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f32_slice_to_f16_avx(src: &[f32], dst: &mut [u16]) {
    let n = src.len().min(dst.len());
    let mut i = 0usize;
    while i + 8 <= n {
        let f = std::arch::x86_64::_mm256_loadu_ps(src.as_ptr().add(i));
        let v = std::arch::x86_64::_mm256_cvtps_ph(f, 0);
        std::arch::x86_64::_mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut _, v);
        i += 8;
    }
    for j in i..n {
        dst[j] = f32_to_f16_bits(src[j]);
    }
}

pub fn f32_slice_to_f16(src: &[f32], dst: &mut [u16]) {
    let n = src.len().min(dst.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 8
        && std::arch::is_x86_feature_detected!("avx")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        unsafe { f32_slice_to_f16_avx(src, dst); }
        return;
    }
    for i in 0..n {
        dst[i] = f32_to_f16_bits(src[i]);
    }
}
