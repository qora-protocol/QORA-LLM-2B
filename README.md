---
license: apache-2.0
language:
  - en
tags:
  - ternary
  - bitnet
  - rust
  - inference
  - cpu
base_model: microsoft/bitnet-b1.58-2B-4T
pipeline_tag: text-generation
---

# QORA-LLM-2B

Pure Rust ternary inference engine based on [BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T). No Python, no CUDA, no external ML frameworks. Single executable + model weights = portable AI that runs on any machine.

**Zero-multiplication inference** — ternary weights {-1, 0, +1} mean the inner GEMV loop uses only addition and subtraction, no floating-point multiply. **Smart system awareness** — detects RAM and CPU at startup and adjusts generation limits automatically.

## License

This project is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). The base model [BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) is released by Microsoft under the MIT license.

## What It Does

QORA-LLM-2B is a 2-billion parameter language model. It can:

- **Text generation** — answer questions, write code, explain concepts
- **Chat mode** — multi-turn conversation with LLaMA 3 chat template
- **Raw mode** — direct text completion without chat formatting

## Architecture

BitNet b1.58 uses a modified transformer with ternary quantized projections and SubLN normalization:

| Component | Details |
|-----------|---------|
| **Parameters** | 2B total |
| **Hidden dim** | 2560 |
| **Layers** | 30 |
| **Attention** | GQA (20 query / 5 KV heads), head_dim=128 |
| **FFN** | 6912 intermediate, Squared ReLU activation |
| **Vocabulary** | 128,256 tokens (LLaMA 3) |
| **Context** | 4096 tokens |
| **RoPE** | rotate_half, theta=500,000 |

### SubLN Pattern (4 norms per layer)

Unlike standard LLaMA (2 norms per layer), BitNet uses SubLN with extra normalization before output projections:

```
residual = x
x = input_layernorm(x)           # RMSNorm [2560]
q, k, v = q/k/v_proj(x)          # Ternary linear (add/sub only)
q, k = apply_rope(q, k)
attn = attention(q, k, v)        # GQA: 20Q/5KV
attn = attn_sub_norm(attn)       # SubLN RMSNorm [2560]
attn = o_proj(attn)              # Ternary linear
x = residual + attn

residual = x
x = post_attention_layernorm(x)  # RMSNorm [2560]
gate = relu2(gate_proj(x))       # Squared ReLU: max(0,x)^2
up = up_proj(x)
x = ffn_sub_norm(gate * up)      # SubLN RMSNorm [6912]
x = down_proj(x)
x = residual + x
```

### Ternary GEMV (No Multiplication)

Each weight is one of {-1, 0, +1}, packed 4 per byte (2 bits each). The inner loop:

```
+1 -> output += input
-1 -> output -= input
 0 -> skip
```

A single scalar multiply by the layer scale factor happens only at the end. This makes BitNet inference fundamentally different from traditional float/quantized models.

## Smart System Awareness

QORA-LLM-2B detects your system at startup and automatically adjusts generation limits:

```
QORA - Native Rust LLM Inference Engine
System: 16101 MB RAM (8271 MB free), 12 threads
```

| Available RAM | Max Tokens | Behavior |
|---------------|------------|----------|
| < 4 GB | 256 (cap 512) | Minimal generation, warning displayed |
| 4-8 GB | 512 (cap 1024) | Constrained, warning displayed |
| 8-12 GB | 1024 (cap 2048) | Normal operation |
| >= 12 GB | 2048 (cap 8192) | Full capability |

Hard caps apply even to explicit user values. Supports **Windows**, **Linux**, and **macOS**.

## Platform Support

| Platform | Binary | Status |
|----------|--------|--------|
| **Windows x86_64** | `qor2b.exe` | Tested |
| **Linux x86_64** | `qor2b` | Supported |
| **macOS aarch64** | `qor2b` | Supported |

## Quick Start

1. Download from the [Releases](https://github.com/qora-protocol/QORA-LLM-2B/releases) page:
   - `model.qor2b` (~1.13 GB)
   - `tokenizer.json`
   - `qor2b.exe` (Windows) or build from source

2. Place all files in the same folder and run:

```bash
# Chat mode (default)
qor2b --prompt "Explain how ternary neural networks work"

# With token limit
qor2b --prompt "Write a haiku about Rust" --max-tokens 100

# Raw text completion (no chat template)
qor2b --prompt "Once upon a time" --raw

# Greedy decoding (deterministic)
qor2b --prompt "What is 2+2?" --greedy
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--prompt TEXT` | Input prompt (default: "Hello, how are you?") |
| `--max-tokens N` | Max tokens to generate (default: auto based on RAM) |
| `--raw` | Raw text completion (skip chat template) |
| `--greedy` | Greedy decoding (temperature=0) |
| `--load PATH` | Custom model path (default: model.qor2b next to exe) |
| `--convert DIR` | Convert safetensors from DIR to .qor2b format |
| `--save PATH` | Output path for conversion (default: model.qor2b) |

### Sampling Defaults

| Parameter | Value |
|-----------|-------|
| temperature | 0.7 |
| top_k | 40 |
| top_p | 0.95 |
| repetition_penalty | 1.1 |
| presence_penalty | 0.6 |

## Converting from Safetensors

To convert the original bf16 weights yourself:

```bash
# Download the model from HuggingFace
# (requires: pip install huggingface_hub)
python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/bitnet-b1.58-2B-4T-bf16', local_dir='bitnet-bf16')"

# Convert to .qor2b format
qor2b --convert bitnet-bf16 --save model.qor2b
```

Conversion takes ~2 minutes and compresses 4.8 GB bf16 safetensors to a 1.13 GB ternary binary.

## Building from Source

```bash
cargo build --release
```

### Dependencies

- **Language**: Pure Rust (2024 edition)
- `rayon` — Thread pool for parallel GEMV and attention
- `half` — F16 support for embeddings
- `tokenizers` — HuggingFace tokenizer (LLaMA 3)
- `safetensors` — Model conversion from HuggingFace format
- `serde_json` — Config parsing
- **No ML framework** — all matrix ops are hand-written Rust

## File Structure

```
src/
  main.rs       — CLI entry point, argument parsing, smart system
  config.rs     — BitNet model configuration
  gemv.rs       — Ternary GEMV kernel, forward pass, attention, RoPE
  generate.rs   — Text generation loop with sampling
  tokenizer.rs  — LLaMA 3 tokenizer and chat template
  save.rs       — Binary model format (.qor2b) save/load
  convert.rs    — Safetensors bf16 -> ternary .qor2b converter
  system.rs     — System resource detection and smart limits
  lib.rs        — Module exports
```

## Model Binary Format (.qor2b)

Custom binary format for fast loading:

```
Header:   "QR2B" magic + version(u32)
Metadata: layers, vocab, hidden, intermediate, heads, kv_heads,
          head_dim, kv_groups, half_dim, rms_eps
Layers:   30x (7 ternary weights + 4 norm vectors)
Global:   f16 embedding + f32 final norm + f32 RoPE tables
```

### Size Breakdown

| Component | Size |
|-----------|------|
| Embedding (f16) | ~656 MB |
| 30 layers ternary (2-bit) | ~470 MB |
| Norms + RoPE tables | ~3 MB |
| **Total** | **~1.13 GB** |

## Performance

Tested on i5-11500 (6C/12T, AVX-512), 16GB RAM:

| Metric | Value |
|--------|-------|
| Decode speed | ~2.5 tok/s |
| Model load time | ~2s |
| Model size | 1.13 GB |
| RAM usage | ~1.5 GB |

## Comparison with Other QORA Models

| | QORA-LLM-2B | QORA-3B | QORA-4B |
|---|-------------|---------|---------|
| Parameters | 2B | 3B | 4B |
| Quantization | Ternary (1.58-bit) | Q4 (4-bit) | Q4 (4-bit) |
| Model size | 1.13 GB | 1.7 GB | 3.5 GB |
| Decode (CPU) | ~2.5 tok/s | ~0.86 tok/s | ~1.3 tok/s |
| RAM usage | ~1.5 GB | ~2.5 GB | ~3.5 GB |
| GPU support | No | Yes | Yes |
| Vision | No | No | Yes |
| Best for | Fast CPU inference, low RAM | General text | Multimodal, reasoning |
