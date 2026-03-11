use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse flags
    let raw_mode = args.iter().any(|a| a == "--raw");
    let greedy = args.iter().any(|a| a == "--greedy");

    // Track whether user explicitly set max-tokens
    let mut max_tokens_explicit = false;

    // Parse key-value arguments
    let mut prompt = String::from("Hello, how are you?");
    let mut max_tokens: usize = 1024;
    let exe_dir = std::env::current_exe()
        .expect("Cannot determine executable path")
        .parent().unwrap().to_path_buf();
    let mut load_path = exe_dir.join("model.qor2b");
    let mut convert_path: Option<PathBuf> = None;
    let mut save_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                if i + 1 < args.len() { prompt = args[i + 1].clone(); i += 1; }
            }
            "--max-tokens" => {
                if i + 1 < args.len() {
                    max_tokens = args[i + 1].parse().unwrap_or(1024);
                    max_tokens_explicit = true;
                    i += 1;
                }
            }
            "--load" => {
                if i + 1 < args.len() { load_path = PathBuf::from(&args[i + 1]); i += 1; }
            }
            "--convert" => {
                if i + 1 < args.len() { convert_path = Some(PathBuf::from(&args[i + 1])); i += 1; }
            }
            "--save" => {
                if i + 1 < args.len() { save_path = Some(PathBuf::from(&args[i + 1])); i += 1; }
            }
            _ => {}
        }
        i += 1;
    }

    // System awareness
    let sys_info = qor2b::system::SystemInfo::detect();
    let limits = sys_info.smart_limits();
    eprintln!("QORA - Native Rust LLM Inference Engine");
    eprintln!("System: {} MB RAM ({} MB free), {} threads",
        sys_info.total_ram_mb, sys_info.available_ram_mb, sys_info.cpu_threads);

    // Set defaults if user didn't specify
    if !max_tokens_explicit {
        max_tokens = limits.default_max_tokens;
    }

    // Hard cap: even explicit values get clamped on weak systems
    if max_tokens > limits.max_tokens {
        eprintln!("System cap: max-tokens {} → {}", max_tokens, limits.max_tokens);
        max_tokens = limits.max_tokens;
    }

    if let Some(warning) = limits.warning {
        eprintln!("WARNING: {warning}");
    }

    // === Convert mode ===
    if let Some(ref model_dir) = convert_path {
        eprintln!("Converting safetensors from {}...", model_dir.display());
        let t0 = Instant::now();
        let weights = qor2b::convert::convert_safetensors(model_dir)
            .expect("Failed to convert model");
        eprintln!("Conversion took {:.1?}", t0.elapsed());

        let out_path = save_path.unwrap_or_else(|| PathBuf::from("model.qor2b"));
        eprintln!("Saving to {}...", out_path.display());
        let t0 = Instant::now();
        qor2b::save::save_model(&weights, &out_path)
            .expect("Failed to save model");
        let file_size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("Saved in {:.1?} ({:.1} MB)", t0.elapsed(), file_size as f64 / (1024.0 * 1024.0));
        return;
    }

    // === Inference mode ===
    let mode_str = if raw_mode { "raw" } else { "chat" };
    eprintln!("Prompt: {prompt}");
    eprintln!("Mode: {mode_str}, max_tokens: {max_tokens}");

    // Load model
    eprintln!("Loading model from {}...", load_path.display());
    let t0 = Instant::now();
    let weights = qor2b::save::load_model(&load_path)
        .expect("Failed to load .qor2b model");
    let mem_mb = weights.memory_bytes() / (1024 * 1024);
    eprintln!("Model loaded in {:.1?} ({mem_mb} MB)", t0.elapsed());

    // Decode-only benchmark (warmup)
    {
        let num_kv_heads = weights.num_kv_heads();
        let head_dim = weights.head_dim();
        let num_layers = weights.num_layers();

        eprintln!("Decode-only benchmark (ternary)...");
        let mut kv_cache = qor2b::gemv::empty_kv_cache(num_layers, num_kv_heads, head_dim);
        let t = Instant::now();
        let _logits = qor2b::gemv::forward_decode_raw(&weights, 1, &mut kv_cache);
        eprintln!("  forward_decode_raw: {:.1?}", t.elapsed());

        let mut kv_cache2 = qor2b::gemv::empty_kv_cache(num_layers, num_kv_heads, head_dim);
        let t = Instant::now();
        let _logits = qor2b::gemv::forward_decode_raw(&weights, 1, &mut kv_cache2);
        eprintln!("  forward_decode_raw (warm): {:.1?}", t.elapsed());
    }

    // Load tokenizer
    let tokenizer_path = load_path.parent()
        .unwrap_or(std::path::Path::new("."))
        .join("tokenizer.json");
    let tokenizer = qor2b::tokenizer::QoraTokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");

    let temperature = if greedy { 0.0 } else { 0.7 };

    let params = qor2b::generate::GenerateParams {
        max_new_tokens: max_tokens,
        eos_token_id: qor2b::tokenizer::EOS,
        eot_token_id: qor2b::tokenizer::EOT,
        temperature,
        top_p: 0.95,
        top_k: 40,
        repetition_penalty: 1.1,
        presence_penalty: 0.6,
    };

    // Generate
    if raw_mode {
        qor2b::generate::generate_raw(&weights, &tokenizer, &prompt, &params);
    } else {
        qor2b::generate::generate(&weights, &tokenizer, &prompt, &params);
    }
}
