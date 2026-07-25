# Bugs Found and Fixed

## Fixed

### 1. GQA K/V Output Offsets (wrong by ×2)
**File:** `inference.rs` lines 558, 565
**Effect:** K and V projection outputs written INSIDE Q's output region instead of after Q. kv_write shader reads K/V from contiguous positions (element 4096/4608) but they were at element 2048/2304. GQA attention used uninitialized memory for K and V (zero if driver zeroes UMA, noise otherwise). GQA layers effectively non-functional.
**Fix:** `+ 4096` → `+ 4096 * 2`, `+ 4608` → `+ 4608 * 2`.
**Note:** Fused shader (`rms_norm_qkv_rope`) uses correct contiguous layout (5120-element stride, K at element 4096). Only unfused fallback path was broken.

### 2. TUI Edit in Raw Mode (no echo)
**File:** `tui.rs`
**Effect:** `edit_field` ran in raw terminal mode (ICANON/ECHO off). User typed blindly — no visible input, no backspace.
**Fix:** Restore cooked mode via `tcsetattr(orig)` before reading input line, re-enter raw mode after.

### 3. Escape Sequence Desync
**File:** `tui.rs`
**Effect:** Non-arrow escape sequences (F1, Home, etc.) left stray bytes on stdin after `\x1B[` was partially consumed. Subsequent `read_key()` calls would read garbage bytes.
**Fix:** After detecting incomplete escape sequence, consume all remaining bytes with `poll()` (50ms timeout each).

### 4. Chat Mode Missing BOS
**File:** `main.rs` (chat_loop)
**Effect:** Turn 2+ conversation encoding didn't prepend BOS. Encoded token sequence was 1 shorter than `state.seq_len`. `generate_mtp` skipped prefill entirely, generating continuation of previous response instead of responding to new user input.
**Fix:** Prepend BOS same as initial encoding in `run_inference`.

### 5. DeltaNet State Double-Offset
**File:** `inference.rs` (original code, fixed in early refactor)
**Effect:** Both Rust and shader added `layer * LAYER_SIZE` to state base. Layer N accessed state at `base + 2 * N * LAYER_SIZE` — past the 30-layer allocation for N > 15. Buffer overrun into adjacent arena regions.
**Fix:** Rust no longer adds layer offset. Shader handles it via `pc.layer_idx`.

### 6. VkBuffer arena_buffer Leak
**File:** `main.rs`
**Effect:** `create_buffer_from_memory` created VkBuffer handle. Never destroyed via `destroy_buffer`. Minor handle leak.
**Fix:** Added `destroy_buffer` before each `free_memory` in all exit paths.

## Known Pre-Existing (not in scope)

### 7. DeltaNet Gate/V Past Buffer
**File:** `deltanet_step.comp` lines 48, 60
**Effect:** Step shader reads V from element offset 4096+ and gate from element 8192+. QKV buffer has 4128 elements only (8256 bytes). V heads 1-31 read past buffer. Gate always reads from uninitialized memory (~20K bytes past buffer).
**Why it works:** Gate reading uninitialized memory likely returns 0 (if arena zero-initialized). Gate=0 means state is NOT accumulated: `S = 0 * S + 1 * outer(k, v) = outer(k, v)`. DeltaNet behaves as per-token MLP instead of recurrent state update. Model relies on feedforward path only.
**Not fixed:** Deep design issue in DeltaNet step shader — requires rethinking QKV layout or stride constants.

### 8. Various `unwrap()` on Vulkan Calls
**Files:** `inference.rs`, `server.rs`
**Effect:** `reset_command_buffer`, `begin_command_buffer`, `end_command_buffer`, `Mutex::lock` called with `unwrap()`. Panics on Vulkan/driver error.
**Accepted:** Development inference engine. Vulkan command recording errors are fatal anyway.
