---
type: concept
title: DeltaNet Gate Fix
date: 2026-07-26
bundle: moe-680m-docs
concepts: [architecture, shaders]
---

# DeltaNet Gate Fix

## Problem

The DeltaNet step shader (`deltanet_step.comp`) read V and gate from positions past the QKV buffer:

| Read | Expected offset (elements) | Buffer has (4128 total) | Status |
|---|---|---|---|
| Q (16×128) | 0 | 0..2047 | ✅ |
| K (16×128) | 2048 | 2048..4095 | ✅ |
| V (32×128) | 4096 | 4096..4127 (32 elems) | ❌ reads past buffer |
| gate[32] | 8192 | — | ❌ way past buffer |

The buffer has only 32 elements after Q+K, which are gate values (1 per V head, 32 V heads). V is not present in this model variant's QKV output.

## Fix

Removed V read (V is not in the buffer). Gate now reads from element 4096 + vh within the buffer. The state update becomes pure decay:

```
S_new = gate · S
```

No outer product term. Without V input, the state decays each step by `gate`. Starting from zeroed initial state, S stays 0. The step shader's output `o = S × q` is 0, so it contributes nothing to the layer output.

The layer output comes entirely from the feedforward path: output projection GEMM (reading Q+K elements 0..4095 from QKV buffer, projecting to hidden size). The residual connection adds the original input.

## Before vs After

| Aspect | Before | After |
|---|---|---|
| V read | Past buffer (UB), garbage values | Removed (not in buffer) |
| Gate read | Past buffer (UB), likely 0 | Reads from element 4096 + vh |
| State update | `gate·S + (1-gate)·outer(k, garbage)` | `gate·S` |
| State value | Uninitialized decay + garbage outer | Clean decay (zero if never updated) |
| Output `o = S×q` | Undefined (from garbaged state) | 0 (state stays 0) |

Overall model behavior unchanged — state was always 0 (initialized zero, never accumulated with V=0 or V=garbage). The fix eliminates undefined behavior (buffer overrun reads).
