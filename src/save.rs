//! Save/load DecodeWeights to a compact binary format (.qor2b).
//!
//! File format:
//!   Header: magic "QR2B" + version(u32)
//!   Metadata: num_layers, vocab, hidden, intermediate, heads, kv_heads,
//!             head_dim, kv_groups, half_dim, rms_eps
//!   Per-layer: 7 ternary weights + 4 norm vectors
//!   Global: f16 embedding + f32 final norm + f32 RoPE tables

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use half::f16;

use crate::gemv::{DecodeWeights, LayerWeightData, TernaryWeightData};

const MAGIC: &[u8; 4] = b"QR2B";
const VERSION: u32 = 1;

// ============================================================
// Low-level I/O helpers
// ============================================================

fn write_u32(w: &mut impl Write, val: u32) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

fn write_u64(w: &mut impl Write, val: u64) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

fn write_f32_val(w: &mut impl Write, val: f32) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

fn write_f32_vec(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    write_u64(w, data.len() as u64)?;
    w.write_all(bytes)
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32_val(r: &mut impl Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_bytes(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_f32_vec(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * 4];
    r.read_exact(&mut bytes)?;
    let data = unsafe {
        let ptr = bytes.as_ptr() as *const f32;
        std::slice::from_raw_parts(ptr, len).to_vec()
    };
    Ok(data)
}

fn read_f16_vec(r: &mut impl Read) -> io::Result<Vec<f16>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * 2];
    r.read_exact(&mut bytes)?;
    let data = unsafe {
        let ptr = bytes.as_ptr() as *const f16;
        std::slice::from_raw_parts(ptr, len).to_vec()
    };
    Ok(data)
}

// ============================================================
// Ternary weight I/O
// ============================================================

fn read_ternary_weight(r: &mut impl Read) -> io::Result<TernaryWeightData> {
    let k = read_u64(r)? as usize;
    let n = read_u64(r)? as usize;
    let scale = read_f32_val(r)?;
    let packed = read_bytes(r)?;
    Ok(TernaryWeightData { packed, scale, k, n })
}

fn read_layer_weights(r: &mut impl Read) -> io::Result<LayerWeightData> {
    let q_proj = read_ternary_weight(r)?;
    let k_proj = read_ternary_weight(r)?;
    let v_proj = read_ternary_weight(r)?;
    let o_proj = read_ternary_weight(r)?;
    let gate_proj = read_ternary_weight(r)?;
    let up_proj = read_ternary_weight(r)?;
    let down_proj = read_ternary_weight(r)?;
    let input_norm = read_f32_vec(r)?;
    let attn_sub_norm = read_f32_vec(r)?;
    let post_attn_norm = read_f32_vec(r)?;
    let ffn_sub_norm = read_f32_vec(r)?;
    Ok(LayerWeightData {
        q_proj, k_proj, v_proj, o_proj,
        gate_proj, up_proj, down_proj,
        input_norm, attn_sub_norm, post_attn_norm, ffn_sub_norm,
    })
}

// ============================================================
// Public save/load API
// ============================================================

/// Save DecodeWeights to a .qor2b binary file.
pub fn save_model(weights: &DecodeWeights, path: &Path) -> io::Result<()> {
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);

    // Header
    w.write_all(MAGIC)?;
    write_u32(&mut w, VERSION)?;

    // Metadata
    write_u32(&mut w, weights.num_layers() as u32)?;
    write_u32(&mut w, weights.vocab() as u32)?;
    write_u32(&mut w, weights.hidden() as u32)?;
    write_u32(&mut w, weights.intermediate() as u32)?;
    write_u32(&mut w, weights.num_heads() as u32)?;
    write_u32(&mut w, weights.num_kv_heads() as u32)?;
    write_u32(&mut w, weights.head_dim() as u32)?;
    write_u32(&mut w, weights.num_kv_groups() as u32)?;
    write_u32(&mut w, weights.half_dim() as u32)?;
    write_f32_val(&mut w, weights.rms_eps())?;

    // Per-layer weights
    for i in 0..weights.num_layers() {
        weights.write_layer(&mut w, i)?;
    }

    // Global: f16 embedding
    weights.write_embed(&mut w)?;

    // Final norm (f32)
    write_f32_vec(&mut w, weights.final_norm_ref())?;

    // RoPE tables (f32)
    write_f32_vec(&mut w, weights.rope_cos_ref())?;
    write_f32_vec(&mut w, weights.rope_sin_ref())?;

    w.flush()?;
    Ok(())
}

/// Load DecodeWeights from a .qor2b binary file.
pub fn load_model(path: &Path) -> io::Result<DecodeWeights> {
    let mut r = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);

    // Header
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("Invalid magic: expected QR2B, got {:?}", std::str::from_utf8(&magic).unwrap_or("???"))));
    }
    let version = read_u32(&mut r)?;
    if version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("Unsupported version {version}, expected {VERSION}")));
    }

    // Metadata
    let num_layers = read_u32(&mut r)? as usize;
    let vocab = read_u32(&mut r)? as usize;
    let hidden = read_u32(&mut r)? as usize;
    let intermediate = read_u32(&mut r)? as usize;
    let num_heads = read_u32(&mut r)? as usize;
    let num_kv_heads = read_u32(&mut r)? as usize;
    let head_dim = read_u32(&mut r)? as usize;
    let num_kv_groups = read_u32(&mut r)? as usize;
    let half_dim = read_u32(&mut r)? as usize;
    let rms_eps = read_f32_val(&mut r)?;

    eprintln!("  BitNet config: {num_layers}L, hidden={hidden}, FFN={intermediate}, \
               heads={num_heads}/{num_kv_heads}, head_dim={head_dim}");

    // Per-layer weights
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if i % 6 == 0 {
            eprintln!("  Loading layer {i}/{num_layers}...");
        }
        layers.push(read_layer_weights(&mut r)?);
    }

    // f16 embedding
    let embed = read_f16_vec(&mut r)?;

    // Final norm
    let final_norm = read_f32_vec(&mut r)?;

    // RoPE tables
    let rope_cos = read_f32_vec(&mut r)?;
    let rope_sin = read_f32_vec(&mut r)?;

    eprintln!("  Loaded ternary model: {num_layers} layers, vocab={vocab}, hidden={hidden}");

    Ok(DecodeWeights::from_parts(
        layers, embed, vocab, hidden, intermediate,
        num_heads, num_kv_heads, head_dim, num_kv_groups,
        final_norm, rope_cos, rope_sin, half_dim, rms_eps,
    ))
}
