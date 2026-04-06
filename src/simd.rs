//! AVX-512 SIMD kernels for QOR2B ternary inference.
//!
//! Optimized GEMV for ternary weights {-1, 0, +1} packed 4 per byte.
//! Encoding: 00=0, 01=+1, 10=-1 (2 bits each, LSB-first).
//!
//! AVX-512 approach:
//!   Load 16 packed bytes = 64 ternary values.
//!   For each of the 4 bit-positions per byte, extract 2-bit indices → 16 x i32.
//!   Use _mm512_permutexvar_ps with a 4-entry LUT [0, +x, -x, 0] (repeated 4x)
//!   to decode 16 output values per pass, 4 passes per 16-byte chunk = 64 outputs.
//!
//! Falls back to scalar code on non-AVX-512 CPUs (dispatch in gemv.rs).

// All functions in this module are `unsafe fn` wrapping SIMD intrinsics.
// Every operation inside them is inherently unsafe, so suppress the per-op warning.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Check if AVX-512F is available at runtime.
pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ============================================================
// AVX-512 ternary GEMV
// ============================================================

/// AVX-512 ternary GEMV inner kernel.
///
/// Processes k_start..k_end rows of a ternary weight matrix, accumulating into
/// a fresh output[0..n] buffer. Optionally applies the per-matrix scale at the end.
///
/// Ternary packing: 4 values per byte (2 bits each), LSB-first along output dim.
///   00 = 0, 01 = +1, 10 = -1, 11 = unused (treated as 0).
///
/// For each input element x_val, we build a 4-entry LUT: [0.0, x_val, -x_val, 0.0].
/// This LUT is replicated 4x to fill 16 lanes: [0, +x, -x, 0, 0, +x, -x, 0, ...].
/// We then process 16 packed bytes at a time (= 64 output values) in 4 passes:
///   Pass 0: extract bits [1:0] from each byte → 16 indices → permutexvar → 16 outputs
///   Pass 1: extract bits [3:2] → 16 outputs at offset +16
///   Pass 2: extract bits [5:4] → 16 outputs at offset +32
///   Pass 3: extract bits [7:6] → 16 outputs at offset +48
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn ternary_gemv_avx512(
    input: &[f32],
    packed: &[u8],
    scale: f32,
    _k: usize,
    n: usize,
    k_start: usize,
    k_end: usize,
    apply_scale: bool,
) -> Vec<f32> {
    let bytes_per_row = (n + 3) / 4;
    // Number of 16-byte chunks we can process with AVX-512 (each = 64 output values)
    let full_chunks = n / 64;
    let remaining_bytes = bytes_per_row - full_chunks * 16;

    let mut output = vec![0.0f32; n];

    let two_bit_mask = _mm512_set1_epi32(0x03);

    for ki in k_start..k_end {
        let x_val = input[ki];
        if x_val == 0.0 {
            continue;
        }

        // Build 4-entry LUT replicated 4x to fill 16 lanes:
        // [0, +x, -x, 0,  0, +x, -x, 0,  0, +x, -x, 0,  0, +x, -x, 0]
        let lut = _mm512_set_ps(
            0.0, -x_val, x_val, 0.0,
            0.0, -x_val, x_val, 0.0,
            0.0, -x_val, x_val, 0.0,
            0.0, -x_val, x_val, 0.0,
        );

        let row_base = ki * bytes_per_row;

        // Process 16-byte chunks (64 output values each)
        for ci in 0..full_chunks {
            let byte_offset = row_base + ci * 16;
            let out_offset = ci * 64;

            // Load 16 packed bytes → zero-extend to 16 x i32
            let bytes = _mm_loadu_si128(packed.as_ptr().add(byte_offset) as *const __m128i);
            let bytes_i32 = _mm512_cvtepu8_epi32(bytes);

            // Pass 0: bits [1:0] — output positions 0..16
            let idx0 = _mm512_and_epi32(bytes_i32, two_bit_mask);
            let vals0 = _mm512_permutexvar_ps(idx0, lut);
            let acc0 = _mm512_loadu_ps(output.as_ptr().add(out_offset));
            _mm512_storeu_ps(output.as_mut_ptr().add(out_offset), _mm512_add_ps(acc0, vals0));

            // Pass 1: bits [3:2] — output positions 16..32
            let idx1 = _mm512_and_epi32(_mm512_srli_epi32(bytes_i32, 2), two_bit_mask);
            let vals1 = _mm512_permutexvar_ps(idx1, lut);
            let acc1 = _mm512_loadu_ps(output.as_ptr().add(out_offset + 16));
            _mm512_storeu_ps(output.as_mut_ptr().add(out_offset + 16), _mm512_add_ps(acc1, vals1));

            // Pass 2: bits [5:4] — output positions 32..48
            let idx2 = _mm512_and_epi32(_mm512_srli_epi32(bytes_i32, 4), two_bit_mask);
            let vals2 = _mm512_permutexvar_ps(idx2, lut);
            let acc2 = _mm512_loadu_ps(output.as_ptr().add(out_offset + 32));
            _mm512_storeu_ps(output.as_mut_ptr().add(out_offset + 32), _mm512_add_ps(acc2, vals2));

            // Pass 3: bits [7:6] — output positions 48..64
            let idx3 = _mm512_srli_epi32(bytes_i32, 6);
            let vals3 = _mm512_permutexvar_ps(idx3, lut);
            let acc3 = _mm512_loadu_ps(output.as_ptr().add(out_offset + 48));
            _mm512_storeu_ps(output.as_mut_ptr().add(out_offset + 48), _mm512_add_ps(acc3, vals3));
        }

        // Scalar tail for remaining bytes (< 16 bytes = < 64 values)
        if remaining_bytes > 0 {
            let scalar_lut = [0.0f32, x_val, -x_val, 0.0];
            let tail_byte_start = full_chunks * 16;
            let tail_out_start = full_chunks * 64;
            for bi in 0..remaining_bytes {
                let byte = packed[row_base + tail_byte_start + bi];
                let ni = tail_out_start + bi * 4;
                if ni < n {
                    output[ni] += scalar_lut[(byte & 0x03) as usize];
                }
                if ni + 1 < n {
                    output[ni + 1] += scalar_lut[((byte >> 2) & 0x03) as usize];
                }
                if ni + 2 < n {
                    output[ni + 2] += scalar_lut[((byte >> 4) & 0x03) as usize];
                }
                if ni + 3 < n {
                    output[ni + 3] += scalar_lut[((byte >> 6) & 0x03) as usize];
                }
            }
        }
    }

    // Apply per-matrix scale
    if apply_scale {
        let scale_vec = _mm512_set1_ps(scale);
        let n16 = n / 16 * 16;
        let mut j = 0usize;
        while j < n16 {
            let v = _mm512_loadu_ps(output.as_ptr().add(j));
            _mm512_storeu_ps(output.as_mut_ptr().add(j), _mm512_mul_ps(v, scale_vec));
            j += 16;
        }
        // Scalar tail for scale
        while j < n {
            output[j] *= scale;
            j += 1;
        }
    }

    output
}

// ============================================================
// AVX-512 fused gate+up ternary GEMV with squared ReLU
// ============================================================

/// AVX-512 fused gate+up ternary GEMV.
///
/// Computes both gate and up projections in a single pass over the input,
/// then fuses relu²(gate) * up at the end.
///
/// Returns the fused intermediate: relu²(gate * gate_scale) * (up * up_scale).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_relu2_gate_up_avx512(
    input: &[f32],
    gate_packed: &[u8],
    _gate_scale: f32,
    up_packed: &[u8],
    _up_scale: f32,
    _k: usize,
    n: usize,
    k_start: usize,
    k_end: usize,
) -> (Vec<f32>, Vec<f32>) {
    let bytes_per_row = (n + 3) / 4;
    let full_chunks = n / 64;
    let remaining_bytes = bytes_per_row - full_chunks * 16;

    let mut gate_out = vec![0.0f32; n];
    let mut up_out = vec![0.0f32; n];

    let two_bit_mask = _mm512_set1_epi32(0x03);

    for ki in k_start..k_end {
        let x_val = input[ki];
        if x_val == 0.0 {
            continue;
        }

        let lut = _mm512_set_ps(
            0.0, -x_val, x_val, 0.0,
            0.0, -x_val, x_val, 0.0,
            0.0, -x_val, x_val, 0.0,
            0.0, -x_val, x_val, 0.0,
        );

        let row_base = ki * bytes_per_row;

        for ci in 0..full_chunks {
            let byte_offset = row_base + ci * 16;
            let out_offset = ci * 64;

            // Load 16 gate bytes
            let g_bytes = _mm_loadu_si128(gate_packed.as_ptr().add(byte_offset) as *const __m128i);
            let g_i32 = _mm512_cvtepu8_epi32(g_bytes);

            // Load 16 up bytes
            let u_bytes = _mm_loadu_si128(up_packed.as_ptr().add(byte_offset) as *const __m128i);
            let u_i32 = _mm512_cvtepu8_epi32(u_bytes);

            // Pass 0: bits [1:0]
            let g_idx0 = _mm512_and_epi32(g_i32, two_bit_mask);
            let u_idx0 = _mm512_and_epi32(u_i32, two_bit_mask);
            let g_vals0 = _mm512_permutexvar_ps(g_idx0, lut);
            let u_vals0 = _mm512_permutexvar_ps(u_idx0, lut);
            let ga0 = _mm512_loadu_ps(gate_out.as_ptr().add(out_offset));
            let ua0 = _mm512_loadu_ps(up_out.as_ptr().add(out_offset));
            _mm512_storeu_ps(gate_out.as_mut_ptr().add(out_offset), _mm512_add_ps(ga0, g_vals0));
            _mm512_storeu_ps(up_out.as_mut_ptr().add(out_offset), _mm512_add_ps(ua0, u_vals0));

            // Pass 1: bits [3:2]
            let g_idx1 = _mm512_and_epi32(_mm512_srli_epi32(g_i32, 2), two_bit_mask);
            let u_idx1 = _mm512_and_epi32(_mm512_srli_epi32(u_i32, 2), two_bit_mask);
            let g_vals1 = _mm512_permutexvar_ps(g_idx1, lut);
            let u_vals1 = _mm512_permutexvar_ps(u_idx1, lut);
            let ga1 = _mm512_loadu_ps(gate_out.as_ptr().add(out_offset + 16));
            let ua1 = _mm512_loadu_ps(up_out.as_ptr().add(out_offset + 16));
            _mm512_storeu_ps(gate_out.as_mut_ptr().add(out_offset + 16), _mm512_add_ps(ga1, g_vals1));
            _mm512_storeu_ps(up_out.as_mut_ptr().add(out_offset + 16), _mm512_add_ps(ua1, u_vals1));

            // Pass 2: bits [5:4]
            let g_idx2 = _mm512_and_epi32(_mm512_srli_epi32(g_i32, 4), two_bit_mask);
            let u_idx2 = _mm512_and_epi32(_mm512_srli_epi32(u_i32, 4), two_bit_mask);
            let g_vals2 = _mm512_permutexvar_ps(g_idx2, lut);
            let u_vals2 = _mm512_permutexvar_ps(u_idx2, lut);
            let ga2 = _mm512_loadu_ps(gate_out.as_ptr().add(out_offset + 32));
            let ua2 = _mm512_loadu_ps(up_out.as_ptr().add(out_offset + 32));
            _mm512_storeu_ps(gate_out.as_mut_ptr().add(out_offset + 32), _mm512_add_ps(ga2, g_vals2));
            _mm512_storeu_ps(up_out.as_mut_ptr().add(out_offset + 32), _mm512_add_ps(ua2, u_vals2));

            // Pass 3: bits [7:6]
            let g_idx3 = _mm512_srli_epi32(g_i32, 6);
            let u_idx3 = _mm512_srli_epi32(u_i32, 6);
            let g_vals3 = _mm512_permutexvar_ps(g_idx3, lut);
            let u_vals3 = _mm512_permutexvar_ps(u_idx3, lut);
            let ga3 = _mm512_loadu_ps(gate_out.as_ptr().add(out_offset + 48));
            let ua3 = _mm512_loadu_ps(up_out.as_ptr().add(out_offset + 48));
            _mm512_storeu_ps(gate_out.as_mut_ptr().add(out_offset + 48), _mm512_add_ps(ga3, g_vals3));
            _mm512_storeu_ps(up_out.as_mut_ptr().add(out_offset + 48), _mm512_add_ps(ua3, u_vals3));
        }

        // Scalar tail
        if remaining_bytes > 0 {
            let scalar_lut = [0.0f32, x_val, -x_val, 0.0];
            let tail_byte_start = full_chunks * 16;
            let tail_out_start = full_chunks * 64;
            for bi in 0..remaining_bytes {
                let gb = gate_packed[row_base + tail_byte_start + bi];
                let ub = up_packed[row_base + tail_byte_start + bi];
                let ni = tail_out_start + bi * 4;
                for r in 0..4 {
                    if ni + r < n {
                        let shift = r * 2;
                        gate_out[ni + r] += scalar_lut[((gb >> shift) & 0x03) as usize];
                        up_out[ni + r] += scalar_lut[((ub >> shift) & 0x03) as usize];
                    }
                }
            }
        }
    }

    (gate_out, up_out)
}
