//! Convert safetensors bf16 weights to ternary .qor2b format.
//!
//! Steps:
//!   1. Load config.json for model architecture
//!   2. Load bf16 safetensors
//!   3. For each projection: transpose, compute scale=mean(|w|), quantize to {-1,0,+1}, pack
//!   4. Keep embedding as f16, norms as f32
//!   5. Precompute RoPE tables
//!   6. Write .qor2b binary

use std::collections::HashMap;
use std::path::Path;

use half::f16;
use safetensors::SafeTensors;

use crate::config::BitNetConfig;
use crate::gemv::{self, DecodeWeights, LayerWeightData, TernaryWeightData};

// ============================================================
// Tensor loading helpers
// ============================================================

struct TensorData {
    data: Vec<u8>,
    shape: Vec<usize>,
}

/// Load all tensors from safetensors files in a directory.
fn load_safetensors_dir(dir: &Path) -> Result<HashMap<String, TensorData>, Box<dyn std::error::Error>> {
    let mut tensors = HashMap::new();

    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().map_or(false, |ext| ext == "safetensors")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    if entries.is_empty() {
        return Err(format!("No .safetensors files found in {}", dir.display()).into());
    }

    for entry in &entries {
        let path = entry.path();
        eprintln!("  Loading {}...", path.file_name().unwrap().to_string_lossy());
        let file_data = std::fs::read(&path)?;
        let st = SafeTensors::deserialize(&file_data)?;
        for (name, view) in st.tensors() {
            tensors.insert(name.to_string(), TensorData {
                data: view.data().to_vec(),
                shape: view.shape().to_vec(),
            });
        }
    }

    eprintln!("  Loaded {} tensors", tensors.len());
    Ok(tensors)
}

/// Convert bf16 raw bytes to f32 values.
fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        let f32_bits = (bits as u32) << 16;
        out.push(f32::from_bits(f32_bits));
    }
    out
}

/// Convert bf16 raw bytes to f16 (half crate).
fn bf16_to_f16(bytes: &[u8]) -> Vec<f16> {
    let f32_vals = bf16_to_f32(bytes);
    f32_vals.iter().map(|&v| f16::from_f32(v)).collect()
}

/// Transpose a matrix from [rows, cols] to [cols, rows] (row-major).
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

// ============================================================
// Ternary quantization
// ============================================================

/// Quantize f32 weights to ternary {-1, 0, +1} and pack 4 per byte.
/// Scale = mean(|w|). Encoding: 00=0, 01=+1, 10=-1.
fn quantize_ternary(weights: &[f32], k: usize, n: usize) -> TernaryWeightData {
    // Compute scale = mean absolute value
    let sum_abs: f64 = weights.iter().map(|w| w.abs() as f64).sum();
    let scale = (sum_abs / weights.len() as f64) as f32;

    let inv_scale = if scale > 1e-10 { 1.0 / scale } else { 0.0 };

    // Quantize
    let mut count_zero = 0usize;
    let mut count_pos = 0usize;
    let mut count_neg = 0usize;

    let bytes_per_row = (n + 3) / 4;
    let mut packed = vec![0u8; k * bytes_per_row];

    for ki in 0..k {
        for ni in 0..n {
            let val = (weights[ki * n + ni] * inv_scale).round().max(-1.0).min(1.0) as i8;
            let encoded = match val {
                1 => { count_pos += 1; 1u8 }
                -1 => { count_neg += 1; 2u8 }
                _ => { count_zero += 1; 0u8 }
            };
            let byte_idx = ki * bytes_per_row + ni / 4;
            let bit_shift = (ni % 4) * 2;
            packed[byte_idx] |= encoded << bit_shift;
        }
    }

    let total = count_zero + count_pos + count_neg;
    eprintln!("    [{k}x{n}] scale={scale:.6}, +1={:.1}%, -1={:.1}%, 0={:.1}%",
        count_pos as f64 / total as f64 * 100.0,
        count_neg as f64 / total as f64 * 100.0,
        count_zero as f64 / total as f64 * 100.0);

    TernaryWeightData { packed, scale, k, n }
}

// ============================================================
// Public conversion API
// ============================================================

/// Convert safetensors model to DecodeWeights.
pub fn convert_safetensors(model_dir: &Path) -> Result<DecodeWeights, Box<dyn std::error::Error>> {
    // 1. Load config
    let config_path = model_dir.join("config.json");
    let config = BitNetConfig::from_file(&config_path)?;
    eprintln!("Config: {:?}", config);

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim();
    let num_kv_groups = config.num_kv_groups();
    let kv_dim = config.kv_dim();

    // 2. Load safetensors
    let tensors = load_safetensors_dir(model_dir)?;

    // Helper: get tensor data as f32 (from bf16)
    let get_f32 = |name: &str| -> Vec<f32> {
        let td = tensors.get(name)
            .unwrap_or_else(|| panic!("Missing tensor: {name}"));
        bf16_to_f32(&td.data)
    };

    let get_shape = |name: &str| -> Vec<usize> {
        tensors.get(name)
            .unwrap_or_else(|| panic!("Missing tensor: {name}"))
            .shape.clone()
    };

    // 3. Load and convert embedding (bf16 → f16)
    eprintln!("Converting embedding...");
    let embed_name = "model.embed_tokens.weight";
    let embed = bf16_to_f16(&tensors.get(embed_name)
        .unwrap_or_else(|| panic!("Missing {embed_name}")).data);
    let embed_shape = get_shape(embed_name);
    eprintln!("  Embedding: {:?} → {} f16 values", embed_shape, embed.len());

    // 4. Convert layers
    let num_layers = config.num_hidden_layers;
    let mut layers = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        eprintln!("Converting layer {i}/{num_layers}...");
        let prefix = format!("model.layers.{i}");

        // Ternary projections: load bf16 → f32, transpose [out,in]→[in,out], quantize
        let convert_proj = |suffix: &str, in_dim: usize, out_dim: usize| -> TernaryWeightData {
            let name = format!("{prefix}.{suffix}");
            let shape = get_shape(&name);
            assert_eq!(shape, vec![out_dim, in_dim],
                "Shape mismatch for {name}: expected [{out_dim},{in_dim}], got {shape:?}");
            let raw = get_f32(&name);
            let transposed = transpose(&raw, out_dim, in_dim); // [out,in] → [in,out]
            quantize_ternary(&transposed, in_dim, out_dim)
        };

        let q_proj = convert_proj("self_attn.q_proj.weight", hidden, num_heads * head_dim);
        let k_proj = convert_proj("self_attn.k_proj.weight", hidden, kv_dim);
        let v_proj = convert_proj("self_attn.v_proj.weight", hidden, kv_dim);
        let o_proj = convert_proj("self_attn.o_proj.weight", num_heads * head_dim, hidden);
        let gate_proj = convert_proj("mlp.gate_proj.weight", hidden, intermediate);
        let up_proj = convert_proj("mlp.up_proj.weight", hidden, intermediate);
        let down_proj = convert_proj("mlp.down_proj.weight", intermediate, hidden);

        // Norms (f32)
        let input_norm = get_f32(&format!("{prefix}.input_layernorm.weight"));
        let attn_sub_norm = get_f32(&format!("{prefix}.self_attn.attn_sub_norm.weight"));
        let post_attn_norm = get_f32(&format!("{prefix}.post_attention_layernorm.weight"));
        let ffn_sub_norm = get_f32(&format!("{prefix}.mlp.ffn_sub_norm.weight"));

        layers.push(LayerWeightData {
            q_proj, k_proj, v_proj, o_proj,
            gate_proj, up_proj, down_proj,
            input_norm, attn_sub_norm, post_attn_norm, ffn_sub_norm,
        });
    }

    // 5. Final norm
    let final_norm = get_f32("model.norm.weight");

    // 6. RoPE tables
    let max_seq = config.max_position_embeddings;
    eprintln!("Precomputing RoPE tables (theta={}, max_seq={max_seq})...", config.rope_theta);
    let (rope_cos, rope_sin, half_dim) =
        gemv::precompute_rope(config.rope_theta, head_dim, max_seq);

    // 7. Build DecodeWeights
    let weights = DecodeWeights::from_parts(
        layers,
        embed,
        config.vocab_size,
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
        config.rms_norm_eps as f32,
    );

    let mem_mb = weights.memory_bytes() / (1024 * 1024);
    eprintln!("Conversion complete: {mem_mb} MB total");

    Ok(weights)
}
