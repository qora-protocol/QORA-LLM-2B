//! Ternary GEMV engine for BitNet b1.58-2B inference.
//!
//! Weights are ternary {-1, 0, +1} packed 4 values per byte (2 bits each).
//! Encoding: 00=0, 01=+1, 10=-1, 11=unused.
//!
//! The inner GEMV loop uses NO multiplication — just add/sub.
//! A single per-matrix scale factor is applied at the end.
//!
//! SubLN architecture: 4 norms per layer (input, attn_sub, post_attn, ffn_sub).
//! Activation: Squared ReLU (relu²) instead of SiLU.
//! RoPE: rotate_half style, theta=500000, all layers.

use half::f16;
use rayon::prelude::*;
use std::sync::OnceLock;

use crate::simd;

/// Cached AVX-512 detection (checked once, reused forever).
fn use_avx512() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        let avx = simd::has_avx512();
        if avx {
            eprintln!("[QOR2B] AVX-512 detected — using SIMD ternary kernels");
        }
        avx
    })
}

// ============================================================
// Weight types
// ============================================================

/// Ternary weight matrix {-1, 0, +1} packed 4 per byte.
/// Encoding: 00=0, 01=+1, 10=-1. Packed LSB-first along output dim.
/// Scale: mean(|original_weight|), one scalar per matrix.
pub(crate) struct TernaryWeight {
    packed: Vec<u8>,
    scale: f32,
    k: usize, // input dim
    n: usize, // output dim
}

impl TernaryWeight {
    fn memory_bytes(&self) -> usize {
        self.packed.len() + 4 // packed + scale
    }
}

// ============================================================
// Public data types for serialization (used by save.rs, convert.rs)
// ============================================================

pub struct TernaryWeightData {
    pub packed: Vec<u8>,
    pub scale: f32,
    pub k: usize,
    pub n: usize,
}

pub struct LayerWeightData {
    pub q_proj: TernaryWeightData,
    pub k_proj: TernaryWeightData,
    pub v_proj: TernaryWeightData,
    pub o_proj: TernaryWeightData,
    pub gate_proj: TernaryWeightData,
    pub up_proj: TernaryWeightData,
    pub down_proj: TernaryWeightData,
    pub input_norm: Vec<f32>,
    pub attn_sub_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub ffn_sub_norm: Vec<f32>,
}

impl From<TernaryWeightData> for TernaryWeight {
    fn from(d: TernaryWeightData) -> Self {
        TernaryWeight {
            packed: d.packed,
            scale: d.scale,
            k: d.k,
            n: d.n,
        }
    }
}

impl From<LayerWeightData> for LayerWeights {
    fn from(ld: LayerWeightData) -> Self {
        LayerWeights {
            q_proj: ld.q_proj.into(),
            k_proj: ld.k_proj.into(),
            v_proj: ld.v_proj.into(),
            o_proj: ld.o_proj.into(),
            gate_proj: ld.gate_proj.into(),
            up_proj: ld.up_proj.into(),
            down_proj: ld.down_proj.into(),
            input_norm: ld.input_norm,
            attn_sub_norm: ld.attn_sub_norm,
            post_attn_norm: ld.post_attn_norm,
            ffn_sub_norm: ld.ffn_sub_norm,
        }
    }
}

// ============================================================
// KV cache
// ============================================================

/// Raw KV cache: per-layer (k_data, v_data, seq_len).
/// Layout: [seq_len, kv_heads, head_dim] token-major, f32.
pub type RawKvCache = Vec<(Vec<f32>, Vec<f32>, usize)>;

pub fn empty_kv_cache(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> RawKvCache {
    (0..num_layers)
        .map(|_| {
            let cap = num_kv_heads * 512 * head_dim;
            (Vec::with_capacity(cap), Vec::with_capacity(cap), 0usize)
        })
        .collect()
}

// ============================================================
// Per-layer and model weight structures
// ============================================================

/// Per-layer weights: 7 ternary projections + 4 RMSNorm (SubLN).
struct LayerWeights {
    q_proj: TernaryWeight,
    k_proj: TernaryWeight,
    v_proj: TernaryWeight,
    o_proj: TernaryWeight,
    gate_proj: TernaryWeight,
    up_proj: TernaryWeight,
    down_proj: TernaryWeight,
    // SubLN: 4 norms per layer
    input_norm: Vec<f32>,       // [hidden]
    attn_sub_norm: Vec<f32>,    // [hidden] — after attention, before o_proj
    post_attn_norm: Vec<f32>,   // [hidden]
    ffn_sub_norm: Vec<f32>,     // [intermediate] — after gate*up, before down_proj
}

/// All model weights for BitNet b1.58-2B inference.
pub struct DecodeWeights {
    layers: Vec<LayerWeights>,
    embed: Vec<f16>,          // [vocab * hidden], f16
    vocab: usize,
    hidden: usize,
    intermediate: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_kv_groups: usize,
    final_norm: Vec<f32>,     // [hidden]
    rope_cos: Vec<f32>,       // [max_seq * half_dim]
    rope_sin: Vec<f32>,
    half_dim: usize,
    rms_eps: f32,
}

impl DecodeWeights {
    pub fn memory_bytes(&self) -> usize {
        let mut total = self.embed.len() * 2; // f16 embed
        total += self.final_norm.len() * 4;
        total += (self.rope_cos.len() + self.rope_sin.len()) * 4;
        for l in &self.layers {
            total += l.q_proj.memory_bytes();
            total += l.k_proj.memory_bytes();
            total += l.v_proj.memory_bytes();
            total += l.o_proj.memory_bytes();
            total += l.gate_proj.memory_bytes();
            total += l.up_proj.memory_bytes();
            total += l.down_proj.memory_bytes();
            total += (l.input_norm.len() + l.attn_sub_norm.len()
                + l.post_attn_norm.len() + l.ffn_sub_norm.len()) * 4;
        }
        total
    }

    // --- Accessors for save.rs ---
    pub fn num_layers(&self) -> usize { self.layers.len() }
    pub fn vocab(&self) -> usize { self.vocab }
    pub fn hidden(&self) -> usize { self.hidden }
    pub fn intermediate(&self) -> usize { self.intermediate }
    pub fn num_heads(&self) -> usize { self.num_heads }
    pub fn num_kv_heads(&self) -> usize { self.num_kv_heads }
    pub fn head_dim(&self) -> usize { self.head_dim }
    pub fn num_kv_groups(&self) -> usize { self.num_kv_groups }
    pub fn half_dim(&self) -> usize { self.half_dim }
    pub fn final_norm_ref(&self) -> &[f32] { &self.final_norm }
    pub fn rope_cos_ref(&self) -> &[f32] { &self.rope_cos }
    pub fn rope_sin_ref(&self) -> &[f32] { &self.rope_sin }
    pub fn embed_ref(&self) -> &[f16] { &self.embed }
    pub fn rms_eps(&self) -> f32 { self.rms_eps }

    /// Write ternary weight to stream.
    pub(crate) fn write_ternary(w: &mut impl std::io::Write, tw: &TernaryWeight) -> std::io::Result<()> {
        w.write_all(&(tw.k as u64).to_le_bytes())?;
        w.write_all(&(tw.n as u64).to_le_bytes())?;
        w.write_all(&tw.scale.to_le_bytes())?;
        w.write_all(&(tw.packed.len() as u64).to_le_bytes())?;
        w.write_all(&tw.packed)?;
        Ok(())
    }

    /// Write all weights for layer i.
    pub fn write_layer(&self, w: &mut impl std::io::Write, i: usize) -> std::io::Result<()> {
        let lw = &self.layers[i];
        Self::write_ternary(w, &lw.q_proj)?;
        Self::write_ternary(w, &lw.k_proj)?;
        Self::write_ternary(w, &lw.v_proj)?;
        Self::write_ternary(w, &lw.o_proj)?;
        Self::write_ternary(w, &lw.gate_proj)?;
        Self::write_ternary(w, &lw.up_proj)?;
        Self::write_ternary(w, &lw.down_proj)?;
        write_f32_vec_io(w, &lw.input_norm)?;
        write_f32_vec_io(w, &lw.attn_sub_norm)?;
        write_f32_vec_io(w, &lw.post_attn_norm)?;
        write_f32_vec_io(w, &lw.ffn_sub_norm)?;
        Ok(())
    }

    /// Write the f16 embedding.
    pub fn write_embed(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(self.embed.as_ptr() as *const u8, self.embed.len() * 2)
        };
        w.write_all(&(self.embed.len() as u64).to_le_bytes())?;
        w.write_all(bytes)?;
        Ok(())
    }

    /// Construct DecodeWeights from deserialized parts.
    pub fn from_parts(
        layer_data: Vec<LayerWeightData>,
        embed: Vec<f16>,
        vocab: usize,
        hidden: usize,
        intermediate: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_kv_groups: usize,
        final_norm: Vec<f32>,
        rope_cos: Vec<f32>,
        rope_sin: Vec<f32>,
        half_dim: usize,
        rms_eps: f32,
    ) -> Self {
        let layers: Vec<LayerWeights> = layer_data.into_iter().map(|ld| ld.into()).collect();
        Self {
            layers,
            embed,
            vocab,
            hidden,
            intermediate,
            num_heads,
            num_kv_heads,
            head_dim,
            num_kv_groups,
            final_norm,
            rope_cos,
            rope_sin,
            half_dim,
            rms_eps,
        }
    }
}

// I/O helpers
fn write_f32_vec_io(w: &mut impl std::io::Write, data: &[f32]) -> std::io::Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    w.write_all(&(data.len() as u64).to_le_bytes())?;
    w.write_all(bytes)
}

// ============================================================
// Ternary GEMV — NO multiplication in inner loop
// ============================================================

/// Single-threaded ternary GEMV for k_start..k_end.
/// Uses LUT: [0.0, +x, -x, 0.0] for branchless decode.
/// Dispatches to AVX-512 SIMD kernel when available.
#[inline]
fn ternary_gemv_inner(
    input: &[f32],
    packed: &[u8],
    scale: f32,
    _k: usize,
    n: usize,
    k_start: usize,
    k_end: usize,
    apply_scale: bool,
) -> Vec<f32> {
    // AVX-512 fast path
    #[cfg(target_arch = "x86_64")]
    if use_avx512() {
        return unsafe {
            simd::ternary_gemv_avx512(input, packed, scale, _k, n, k_start, k_end, apply_scale)
        };
    }

    // Scalar fallback
    let bytes_per_row = (n + 3) / 4;
    let full_quads = n / 4;
    let remainder = n % 4;
    let mut output = vec![0.0f32; n];

    for ki in k_start..k_end {
        let x_val = input[ki];
        if x_val == 0.0 { continue; }

        let lut = [0.0f32, x_val, -x_val, 0.0];
        let row_base = ki * bytes_per_row;

        // Process 4 values per byte (unrolled)
        for bi in 0..full_quads {
            let byte = packed[row_base + bi];
            let ni = bi * 4;
            output[ni]     += lut[(byte & 0x03) as usize];
            output[ni + 1] += lut[((byte >> 2) & 0x03) as usize];
            output[ni + 2] += lut[((byte >> 4) & 0x03) as usize];
            output[ni + 3] += lut[((byte >> 6) & 0x03) as usize];
        }

        // Handle remainder
        if remainder > 0 {
            let byte = packed[row_base + full_quads];
            let ni = full_quads * 4;
            for r in 0..remainder {
                output[ni + r] += lut[((byte >> (r * 2)) & 0x03) as usize];
            }
        }
    }

    // Apply per-matrix scale
    if apply_scale {
        for v in output.iter_mut() {
            *v *= scale;
        }
    }
    output
}

/// Ternary GEMV — multi-threaded for large matrices.
#[inline]
fn ternary_gemv(input: &[f32], weight: &TernaryWeight) -> Vec<f32> {
    let k = weight.k;
    let n = weight.n;

    // Small matrices: single-threaded
    if k * n < 2_000_000 {
        return ternary_gemv_inner(input, &weight.packed, weight.scale, k, n, 0, k, true);
    }

    // Large matrices: parallel across k dimension
    let num_threads = rayon::current_num_threads();
    let chunk_k = (k + num_threads - 1) / num_threads;

    let partials: Vec<Vec<f32>> = (0..num_threads)
        .into_par_iter()
        .filter_map(|t| {
            let k_start = t * chunk_k;
            let k_end = ((t + 1) * chunk_k).min(k);
            if k_start >= k { return None; }
            Some(ternary_gemv_inner(input, &weight.packed, weight.scale, k, n, k_start, k_end, false))
        })
        .collect();

    let mut output = vec![0.0f32; n];
    for partial in &partials {
        for j in 0..n {
            output[j] += partial[j];
        }
    }
    // Apply scale once after summing
    let scale = weight.scale;
    for v in output.iter_mut() {
        *v *= scale;
    }
    output
}

/// Fused gate+up ternary GEMV: single input read, dual accumulation.
/// Computes relu²(gate(x)) * up(x) with fused projections.
/// Dispatches to AVX-512 SIMD kernel when available.
fn fused_relu2_gate_up(input: &[f32], gate_w: &TernaryWeight, up_w: &TernaryWeight) -> Vec<f32> {
    let k = gate_w.k;
    let n = gate_w.n;

    let num_threads = rayon::current_num_threads();
    let chunk_k = (k + num_threads - 1) / num_threads;

    let partials: Vec<(Vec<f32>, Vec<f32>)> = (0..num_threads)
        .into_par_iter()
        .filter_map(|t| {
            let k_start = t * chunk_k;
            let k_end = ((t + 1) * chunk_k).min(k);
            if k_start >= k { return None; }

            // AVX-512 fast path
            #[cfg(target_arch = "x86_64")]
            if use_avx512() {
                return Some(unsafe {
                    simd::fused_relu2_gate_up_avx512(
                        input, &gate_w.packed, gate_w.scale,
                        &up_w.packed, up_w.scale, k, n, k_start, k_end,
                    )
                });
            }

            // Scalar fallback
            let bytes_per_row = (n + 3) / 4;
            let full_quads = n / 4;
            let remainder = n % 4;

            let mut gate_out = vec![0.0f32; n];
            let mut up_out = vec![0.0f32; n];

            for ki in k_start..k_end {
                let x_val = input[ki];
                if x_val == 0.0 { continue; }

                let lut = [0.0f32, x_val, -x_val, 0.0];
                let row_base = ki * bytes_per_row;

                for bi in 0..full_quads {
                    let gb = gate_w.packed[row_base + bi];
                    let ub = up_w.packed[row_base + bi];
                    let ni = bi * 4;

                    gate_out[ni]     += lut[(gb & 0x03) as usize];
                    gate_out[ni + 1] += lut[((gb >> 2) & 0x03) as usize];
                    gate_out[ni + 2] += lut[((gb >> 4) & 0x03) as usize];
                    gate_out[ni + 3] += lut[((gb >> 6) & 0x03) as usize];

                    up_out[ni]     += lut[(ub & 0x03) as usize];
                    up_out[ni + 1] += lut[((ub >> 2) & 0x03) as usize];
                    up_out[ni + 2] += lut[((ub >> 4) & 0x03) as usize];
                    up_out[ni + 3] += lut[((ub >> 6) & 0x03) as usize];
                }

                if remainder > 0 {
                    let gb = gate_w.packed[row_base + full_quads];
                    let ub = up_w.packed[row_base + full_quads];
                    let ni = full_quads * 4;
                    for r in 0..remainder {
                        let shift = r * 2;
                        gate_out[ni + r] += lut[((gb >> shift) & 0x03) as usize];
                        up_out[ni + r] += lut[((ub >> shift) & 0x03) as usize];
                    }
                }
            }
            Some((gate_out, up_out))
        })
        .collect();

    // Sum partials, apply scales, and fuse relu² * up
    let gate_scale = gate_w.scale;
    let up_scale = up_w.scale;
    let mut gate_final = vec![0.0f32; n];
    let mut up_final = vec![0.0f32; n];
    for (gp, up) in &partials {
        for j in 0..n {
            gate_final[j] += gp[j];
            up_final[j] += up[j];
        }
    }

    let mut out = vec![0.0f32; n];
    for j in 0..n {
        let g = gate_final[j] * gate_scale;
        let relu2 = if g > 0.0 { g * g } else { 0.0 };
        out[j] = relu2 * (up_final[j] * up_scale);
    }
    out
}

/// Ternary GEMM: [seq_len, k] @ [k, n] -> [seq_len, n].
#[inline]
fn ternary_gemm(x: &[f32], seq_len: usize, weight: &TernaryWeight) -> Vec<f32> {
    let k = weight.k;
    let n = weight.n;

    if seq_len <= 1 {
        let row = ternary_gemv_inner(
            &x[..k], &weight.packed, weight.scale, k, n, 0, k, true,
        );
        return row;
    }

    // Parallel across tokens
    let mut output = vec![0.0f32; seq_len * n];
    output.par_chunks_mut(n)
        .enumerate()
        .for_each(|(t, out_row)| {
            let x_row = &x[t * k..(t + 1) * k];
            let row = ternary_gemv_inner(
                x_row, &weight.packed, weight.scale, k, n, 0, k, true,
            );
            out_row.copy_from_slice(&row);
        });
    output
}

// ============================================================
// Embedding lookup (f16)
// ============================================================

#[inline]
fn embed_lookup(embed: &[f16], token_id: usize, hidden: usize) -> Vec<f32> {
    let start = token_id * hidden;
    embed[start..start + hidden]
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

// ============================================================
// Shared compute kernels
// ============================================================

/// RmsNorm with f32 gamma. Standard BitNet: weight * x * inv_rms (NOT 1+weight).
#[inline]
fn rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let size = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / size as f32 + eps).sqrt();
    let mut out = vec![0.0f32; size];
    for i in 0..size {
        out[i] = x[i] * inv_rms * gamma[i];
    }
    out
}

/// Apply RoPE in-place (rotate_half / split-half style).
/// Pairs: (x[base+i], x[base+half+i]) for i in 0..half_dim.
#[inline]
fn apply_rope(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
    half_dim: usize,
    position: usize,
) {
    let cos_offset = position * half_dim;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half_dim {
            let x1 = data[base + i];
            let x2 = data[base + half_dim + i];
            let c = cos_table[cos_offset + i];
            let s = sin_table[cos_offset + i];
            data[base + i] = x1 * c - x2 * s;
            data[base + half_dim + i] = x2 * c + x1 * s;
        }
    }
}

/// In-place softmax.
#[inline]
fn softmax(scores: &mut [f32]) {
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    let inv_sum = 1.0 / sum;
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }
}

/// Parallel lm_head with f16 embedding (tie_word_embeddings).
#[inline(never)]
fn lm_head_parallel(input: &[f32], embed: &[f16], vocab: usize, hidden: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; vocab];
    output.par_chunks_mut(256)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * 256;
            for (i, out) in out_chunk.iter_mut().enumerate() {
                let r = start + i;
                let row = &embed[r * hidden..(r + 1) * hidden];
                let mut sum = 0.0f32;
                for j in 0..hidden {
                    sum += input[j] * row[j].to_f32();
                }
                *out = sum;
            }
        });
    output
}

// ============================================================
// Forward decode (single token)
// ============================================================

/// Fast single-token decode for BitNet b1.58 with SubLN.
///
/// Layer flow:
///   residual = x
///   x = input_norm(x)
///   q, k, v = ternary projections
///   q, k = rope(q, k)
///   attn = attention(q, k, v)
///   attn = attn_sub_norm(attn)       ← SubLN
///   attn = o_proj(attn)
///   x = residual + attn
///
///   residual = x
///   x = post_attn_norm(x)
///   gate = relu²(gate_proj(x))
///   up = up_proj(x)
///   x = ffn_sub_norm(gate * up)      ← SubLN
///   x = down_proj(x)
///   x = residual + x
pub fn forward_decode_raw(
    weights: &DecodeWeights,
    token_id: usize,
    kv_cache: &mut RawKvCache,
) -> Vec<f32> {
    let hidden = weights.hidden;
    let num_layers = weights.layers.len();
    let eps = weights.rms_eps;

    // Token embedding
    let mut x = embed_lookup(&weights.embed, token_id, hidden);

    for i in 0..num_layers {
        let lw = &weights.layers[i];
        let (ref mut cached_k, ref mut cached_v, ref mut cached_len) = kv_cache[i];
        let offset = *cached_len;

        let num_heads = weights.num_heads;
        let num_kv_heads = weights.num_kv_heads;
        let head_dim = weights.head_dim;
        let num_kv_groups = weights.num_kv_groups;

        // === Attention block ===
        // Pre-attention RmsNorm
        let x_norm = rms_norm(&x, &lw.input_norm, eps);

        // QKV projections (ternary)
        let mut q = ternary_gemv(&x_norm, &lw.q_proj);
        let mut k_new = ternary_gemv(&x_norm, &lw.k_proj);
        let v_new = ternary_gemv(&x_norm, &lw.v_proj);

        // Apply RoPE (all layers)
        apply_rope(&mut q, num_heads, head_dim,
            &weights.rope_cos, &weights.rope_sin, weights.half_dim, offset);
        apply_rope(&mut k_new, num_kv_heads, head_dim,
            &weights.rope_cos, &weights.rope_sin, weights.half_dim, offset);

        // Append to KV cache
        cached_k.extend_from_slice(&k_new);
        cached_v.extend_from_slice(&v_new);
        *cached_len = offset + 1;
        let kv_seq_len = *cached_len;

        // Attention (GQA)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_stride = num_kv_heads * head_dim;

        let attn_output: Vec<f32> = if kv_seq_len >= 64 {
            // Parallel across heads
            let head_results: Vec<Vec<f32>> = (0..num_heads)
                .into_par_iter()
                .map(|h| {
                    let kv_h = h / num_kv_groups;
                    let q_offset = h * head_dim;
                    let q_vec = &q[q_offset..q_offset + head_dim];

                    let mut scores = vec![0.0f32; kv_seq_len];
                    for s in 0..kv_seq_len {
                        let k_offset = s * kv_stride + kv_h * head_dim;
                        let k_vec = &cached_k[k_offset..k_offset + head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim { dot += q_vec[d] * k_vec[d]; }
                        scores[s] = dot * scale;
                    }

                    softmax(&mut scores);

                    let mut head_out = vec![0.0f32; head_dim];
                    for s in 0..kv_seq_len {
                        let v_offset = s * kv_stride + kv_h * head_dim;
                        let score = scores[s];
                        for d in 0..head_dim {
                            head_out[d] += score * cached_v[v_offset + d];
                        }
                    }
                    head_out
                })
                .collect();

            let mut out = Vec::with_capacity(num_heads * head_dim);
            for hr in head_results { out.extend_from_slice(&hr); }
            out
        } else {
            // Short context: serial
            let mut attn_out = vec![0.0f32; num_heads * head_dim];
            for h in 0..num_heads {
                let kv_h = h / num_kv_groups;
                let q_offset = h * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                let mut scores = vec![0.0f32; kv_seq_len];
                for s in 0..kv_seq_len {
                    let k_offset = s * kv_stride + kv_h * head_dim;
                    let k_vec = &cached_k[k_offset..k_offset + head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim { dot += q_vec[d] * k_vec[d]; }
                    scores[s] = dot * scale;
                }

                softmax(&mut scores);

                let out_offset = h * head_dim;
                for s in 0..kv_seq_len {
                    let v_offset = s * kv_stride + kv_h * head_dim;
                    let score = scores[s];
                    for d in 0..head_dim {
                        attn_out[out_offset + d] += score * cached_v[v_offset + d];
                    }
                }
            }
            attn_out
        };

        // SubLN: attn_sub_norm BEFORE o_proj
        let attn_normed = rms_norm(&attn_output, &lw.attn_sub_norm, eps);

        // O projection (ternary)
        let attn_out = ternary_gemv(&attn_normed, &lw.o_proj);

        // Residual
        for j in 0..hidden { x[j] += attn_out[j]; }

        // === MLP block ===
        // Pre-MLP RmsNorm
        let x_norm = rms_norm(&x, &lw.post_attn_norm, eps);

        // Fused: relu²(gate(x)) * up(x)
        let intermediate = fused_relu2_gate_up(&x_norm, &lw.gate_proj, &lw.up_proj);

        // SubLN: ffn_sub_norm BEFORE down_proj
        let inter_normed = rms_norm(&intermediate, &lw.ffn_sub_norm, eps);

        // Down projection (ternary)
        let mlp_out = ternary_gemv(&inter_normed, &lw.down_proj);

        // Residual
        for j in 0..hidden { x[j] += mlp_out[j]; }
    }

    // Final norm
    x = rms_norm(&x, &weights.final_norm, eps);

    // lm_head (f16 embedding, tie_word_embeddings)
    lm_head_parallel(&x, &weights.embed, weights.vocab, weights.hidden)
}

// ============================================================
// Raw prefill (full prompt)
// ============================================================

/// Raw prefill: process entire prompt. Returns (last_token_logits, kv_cache).
pub fn raw_prefill(
    weights: &DecodeWeights,
    token_ids: &[u32],
) -> (Vec<f32>, RawKvCache) {
    let hidden = weights.hidden;
    let seq_len = token_ids.len();
    let num_layers = weights.layers.len();
    let eps = weights.rms_eps;

    // Embedding: [seq_len, hidden]
    let mut x = vec![0.0f32; seq_len * hidden];
    for (t, &tid) in token_ids.iter().enumerate() {
        let row = embed_lookup(&weights.embed, tid as usize, hidden);
        x[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
    }

    let mut kv_cache = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let lw = &weights.layers[i];
        let num_heads = weights.num_heads;
        let num_kv_heads = weights.num_kv_heads;
        let head_dim = weights.head_dim;
        let num_kv_groups = weights.num_kv_groups;

        // Pre-attention RmsNorm per-token
        let mut x_norm = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let row = &x[t * hidden..(t + 1) * hidden];
            let normed = rms_norm(row, &lw.input_norm, eps);
            x_norm[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
        }

        // QKV GEMM
        let mut q_all = ternary_gemm(&x_norm, seq_len, &lw.q_proj);
        let mut k_all = ternary_gemm(&x_norm, seq_len, &lw.k_proj);
        let v_all = ternary_gemm(&x_norm, seq_len, &lw.v_proj);

        // RoPE per-token (all layers)
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        for t in 0..seq_len {
            apply_rope(
                &mut q_all[t * q_dim..(t + 1) * q_dim],
                num_heads, head_dim,
                &weights.rope_cos, &weights.rope_sin, weights.half_dim, t,
            );
            apply_rope(
                &mut k_all[t * kv_dim..(t + 1) * kv_dim],
                num_kv_heads, head_dim,
                &weights.rope_cos, &weights.rope_sin, weights.half_dim, t,
            );
        }

        // KV cache
        let cached_k = k_all.clone();
        let cached_v = v_all.clone();

        // Causal attention — parallel across heads
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q_stride = num_heads * head_dim;
        let kv_stride = num_kv_heads * head_dim;

        let head_results: Vec<Vec<f32>> = (0..num_heads)
            .into_par_iter()
            .map(|h| {
                let kv_h = h / num_kv_groups;
                let mut head_out = vec![0.0f32; seq_len * head_dim];

                for t1 in 0..seq_len {
                    let attend_len = t1 + 1;
                    let q_off = t1 * q_stride + h * head_dim;
                    let q_vec = &q_all[q_off..q_off + head_dim];

                    let mut scores = vec![0.0f32; attend_len];
                    for t2 in 0..attend_len {
                        let k_off = t2 * kv_stride + kv_h * head_dim;
                        let k_vec = &cached_k[k_off..k_off + head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim { dot += q_vec[d] * k_vec[d]; }
                        scores[t2] = dot * scale;
                    }

                    softmax(&mut scores);

                    let out_base = t1 * head_dim;
                    for t2 in 0..attend_len {
                        let v_off = t2 * kv_stride + kv_h * head_dim;
                        let score = scores[t2];
                        for d in 0..head_dim {
                            head_out[out_base + d] += score * cached_v[v_off + d];
                        }
                    }
                }
                head_out
            })
            .collect();

        // Interleave head results: [seq_len, num_heads, head_dim]
        let mut attn_output = vec![0.0f32; seq_len * q_stride];
        for (h, hr) in head_results.iter().enumerate() {
            for t in 0..seq_len {
                let src = &hr[t * head_dim..(t + 1) * head_dim];
                let dst_off = t * q_stride + h * head_dim;
                attn_output[dst_off..dst_off + head_dim].copy_from_slice(src);
            }
        }

        // SubLN: attn_sub_norm per-token BEFORE o_proj
        let mut attn_normed = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let row = &attn_output[t * hidden..(t + 1) * hidden];
            let normed = rms_norm(row, &lw.attn_sub_norm, eps);
            attn_normed[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
        }

        // O projection GEMM
        let o_out = ternary_gemm(&attn_normed, seq_len, &lw.o_proj);

        // Residual
        for j in 0..seq_len * hidden { x[j] += o_out[j]; }

        // Pre-MLP RmsNorm
        let mut x_norm2 = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let row = &x[t * hidden..(t + 1) * hidden];
            let normed = rms_norm(row, &lw.post_attn_norm, eps);
            x_norm2[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
        }

        // MLP: separate GEMMs for prefill
        let gate_all = ternary_gemm(&x_norm2, seq_len, &lw.gate_proj);
        let up_all = ternary_gemm(&x_norm2, seq_len, &lw.up_proj);

        let inter_size = weights.intermediate;
        let mut intermediate = vec![0.0f32; seq_len * inter_size];
        for j in 0..seq_len * inter_size {
            let g = gate_all[j];
            let relu2 = if g > 0.0 { g * g } else { 0.0 };
            intermediate[j] = relu2 * up_all[j];
        }

        // SubLN: ffn_sub_norm per-token BEFORE down_proj
        let mut inter_normed = vec![0.0f32; seq_len * inter_size];
        for t in 0..seq_len {
            let row = &intermediate[t * inter_size..(t + 1) * inter_size];
            let normed = rms_norm(row, &lw.ffn_sub_norm, eps);
            inter_normed[t * inter_size..(t + 1) * inter_size].copy_from_slice(&normed);
        }

        let mlp_out = ternary_gemm(&inter_normed, seq_len, &lw.down_proj);

        // Residual
        for j in 0..seq_len * hidden { x[j] += mlp_out[j]; }

        kv_cache.push((cached_k, cached_v, seq_len));

        if i % 6 == 0 || i == num_layers - 1 {
            eprintln!("  Prefill layer {i}/{num_layers}");
        }
    }

    // Final norm (last token only)
    let last_row = &x[(seq_len - 1) * hidden..seq_len * hidden];
    let normed = rms_norm(last_row, &weights.final_norm, eps);

    // lm_head
    let logits = lm_head_parallel(&normed, &weights.embed, weights.vocab, hidden);

    (logits, kv_cache)
}

/// Precompute RoPE cos/sin tables.
pub fn precompute_rope(theta: f64, head_dim: usize, max_seq: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let half_dim = head_dim / 2;
    let mut cos_table = vec![0.0f32; max_seq * half_dim];
    let mut sin_table = vec![0.0f32; max_seq * half_dim];

    for pos in 0..max_seq {
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            cos_table[pos * half_dim + i] = angle.cos() as f32;
            sin_table[pos * half_dim + i] = angle.sin() as f32;
        }
    }

    (cos_table, sin_table, half_dim)
}
