//! Generation loop: prefill → decode with sampling.
//!
//! Features: top-k/top-p sampling, repetition/presence penalty,
//! loop detection, sentence-boundary stopping.

use std::io::Write;
use std::time::Instant;

use crate::gemv::{self, DecodeWeights};
use crate::tokenizer::QoraTokenizer;

/// Simple xoshiro128+ PRNG.
struct Rng {
    s: [u32; 4],
}

impl Rng {
    fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self {
            s: [
                seed as u32,
                (seed >> 32) as u32,
                seed.wrapping_mul(0x9E3779B97F4A7C15) as u32,
                (seed.wrapping_mul(0x9E3779B97F4A7C15) >> 32) as u32,
            ],
        }
    }

    fn next_u32(&mut self) -> u32 {
        let result = self.s[0].wrapping_add(self.s[3]);
        let t = self.s[1] << 9;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(11);
        result
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Generation parameters.
pub struct GenerateParams {
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
    pub eot_token_id: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
}

/// Apply repetition penalty: reduce logits for tokens already generated.
fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty == 1.0 { return; }
    for &tok in generated {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Apply presence penalty: flat subtraction for any token that appeared.
fn apply_presence_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty == 0.0 { return; }
    let mut seen = std::collections::HashSet::new();
    for &tok in generated {
        if seen.insert(tok) {
            let idx = tok as usize;
            if idx < logits.len() {
                logits[idx] -= penalty;
            }
        }
    }
}

/// Detect repetition loops in generated tokens.
fn is_stuck_in_loop(tokens: &[u32]) -> bool {
    let len = tokens.len();
    if len < 20 { return false; }

    // Check for short loops (period 2-8)
    for period in 2..=8 {
        if len < period * 4 { continue; }
        let mut is_loop = true;
        for i in 0..period * 3 {
            if tokens[len - 1 - i] != tokens[len - 1 - i - period] {
                is_loop = false;
                break;
            }
        }
        if is_loop { return true; }
    }
    false
}

/// Chat generation with prefill + decode loop.
pub fn generate(
    weights: &DecodeWeights,
    tokenizer: &QoraTokenizer,
    prompt: &str,
    params: &GenerateParams,
) -> String {
    let tokens = tokenizer.format_chat(prompt, params.max_new_tokens);
    eprintln!("Chat prompt tokens: {}", tokens.len());

    let mut generated_tokens: Vec<u32> = tokens.clone();
    let mut rng = Rng::new();

    // === Prefill ===
    let t0 = Instant::now();
    let (mut logits, mut kv_cache) = gemv::raw_prefill(weights, &tokens);
    let prefill_time = t0.elapsed();
    eprintln!("Prefill: {} tokens in {prefill_time:.1?}", tokens.len());

    if params.temperature > 0.0 {
        apply_repetition_penalty(&mut logits, &generated_tokens, params.repetition_penalty);
    }
    let mut next_token_id = sample_token_top_k(
        &logits, params.temperature, params.top_p, params.top_k, &mut rng,
    );
    let decode_start = Instant::now();
    let mut decode_tokens = 0u32;

    // === Decode loop ===
    for step in 0..params.max_new_tokens {
        // Stop on EOS or EOT
        if next_token_id == params.eos_token_id || next_token_id == params.eot_token_id {
            eprintln!("\n[EOS/EOT after {step} tokens]");
            break;
        }

        generated_tokens.push(next_token_id);

        let token_text = tokenizer.decode(&[next_token_id]);
        print!("{token_text}");
        std::io::stdout().flush().ok();

        // Sentence-boundary stop near token budget (85%)
        if step >= params.max_new_tokens * 85 / 100 {
            let piece = tokenizer.decode(&[next_token_id]);
            let trimmed = piece.trim_end();
            if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
                eprintln!("\n[clean stop near token limit at step {step}]");
                break;
            }
        }

        // Loop detection
        if is_stuck_in_loop(&generated_tokens) {
            eprintln!("\n[loop detected at step {step}, forcing EOS]");
            break;
        }

        // GEMV decode
        let mut logits = gemv::forward_decode_raw(weights, next_token_id as usize, &mut kv_cache);

        if params.temperature > 0.0 {
            apply_repetition_penalty(&mut logits, &generated_tokens, params.repetition_penalty);
            apply_presence_penalty(&mut logits, &generated_tokens, params.presence_penalty);
        }

        next_token_id = sample_token_top_k(
            &logits, params.temperature, params.top_p, params.top_k, &mut rng,
        );
        decode_tokens += 1;

        if decode_tokens % 50 == 0 {
            let elapsed = decode_start.elapsed();
            let tps = decode_tokens as f64 / elapsed.as_secs_f64();
            eprint!("[{decode_tokens} tokens, {tps:.1} tok/s] ");
            std::io::stderr().flush().ok();
        }
    }
    println!();

    let decode_elapsed = decode_start.elapsed();
    if decode_tokens > 0 {
        eprintln!("Decode: {} tokens in {decode_elapsed:.1?} ({:.2} tok/s)",
            decode_tokens,
            decode_tokens as f64 / decode_elapsed.as_secs_f64());
    }

    tokenizer.decode(&generated_tokens)
}

/// Raw text completion (no chat template).
pub fn generate_raw(
    weights: &DecodeWeights,
    tokenizer: &QoraTokenizer,
    prompt: &str,
    params: &GenerateParams,
) -> String {
    let tokens = tokenizer.encode(prompt);
    eprintln!("Prompt tokens: {}", tokens.len());

    let mut all_tokens = tokens.clone();
    let mut rng = Rng::new();

    // === Prefill ===
    let t0 = Instant::now();
    let (mut logits, mut kv_cache) = gemv::raw_prefill(weights, &tokens);
    let prefill_time = t0.elapsed();
    eprintln!("Prefill: {} tokens in {prefill_time:.1?}", tokens.len());

    if params.temperature > 0.0 {
        apply_repetition_penalty(&mut logits, &all_tokens, params.repetition_penalty);
    }
    let mut next_token_id = sample_token_top_k(
        &logits, params.temperature, params.top_p, params.top_k, &mut rng,
    );
    let decode_start = Instant::now();
    let mut decode_tokens = 0u32;

    // === Decode loop ===
    for step in 0..params.max_new_tokens {
        if next_token_id == params.eos_token_id || next_token_id == params.eot_token_id {
            eprintln!("\n[EOS/EOT after {step} tokens]");
            break;
        }

        all_tokens.push(next_token_id);

        let token_text = tokenizer.decode(&[next_token_id]);
        print!("{token_text}");
        std::io::stdout().flush().ok();

        if is_stuck_in_loop(&all_tokens) {
            eprintln!("\n[loop detected at step {step}, forcing EOS]");
            break;
        }

        let mut logits = gemv::forward_decode_raw(weights, next_token_id as usize, &mut kv_cache);
        if params.temperature > 0.0 {
            apply_repetition_penalty(&mut logits, &all_tokens, params.repetition_penalty);
        }
        next_token_id = sample_token_top_k(
            &logits, params.temperature, params.top_p, params.top_k, &mut rng,
        );
        decode_tokens += 1;

        if decode_tokens % 50 == 0 {
            let elapsed = decode_start.elapsed();
            let tps = decode_tokens as f64 / elapsed.as_secs_f64();
            eprint!("[{decode_tokens} tokens, {tps:.1} tok/s] ");
            std::io::stderr().flush().ok();
        }
    }
    println!();

    let decode_elapsed = decode_start.elapsed();
    if decode_tokens > 0 {
        eprintln!("Decode: {} tokens in {decode_elapsed:.1?} ({:.2} tok/s)",
            decode_tokens,
            decode_tokens as f64 / decode_elapsed.as_secs_f64());
    }

    tokenizer.decode(&all_tokens)
}

/// Sample with top-k + top-p (nucleus) filtering.
fn sample_token_top_k(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_val { max_val = v; max_idx = i; }
        }
        return max_idx as u32;
    }

    let inv_temp = 1.0 / temperature;
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &l)| (i as u32, l))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k = if top_k > 0 { top_k.min(indexed.len()) } else { indexed.len() };
    let top_k_candidates = &indexed[..k];

    let mut probs: Vec<(u32, f32)> = top_k_candidates.iter()
        .map(|&(id, l)| (id, ((l - max_logit) * inv_temp).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum <= 0.0 { return probs[0].0; }
    let inv = 1.0 / sum;
    for (_, p) in probs.iter_mut() { *p *= inv; }

    let mut cumsum = 0.0f32;
    let mut cutoff = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p { cutoff = i + 1; break; }
    }
    let candidates = &probs[..cutoff];
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    if total <= 0.0 { return candidates[0].0; }

    let rand_val = rng.next_f32() * total;
    let mut accum = 0.0f32;
    for &(id, prob) in candidates {
        accum += prob;
        if accum >= rand_val { return id; }
    }
    candidates[0].0
}
