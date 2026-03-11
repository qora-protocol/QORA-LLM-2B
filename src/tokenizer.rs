use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

// LLaMA 3 special token IDs
pub const BOS: u32 = 128000;      // <|begin_of_text|>
pub const EOS: u32 = 128001;      // <|end_of_text|>
pub const EOT: u32 = 128009;      // <|eot_id|>

pub struct QoraTokenizer {
    inner: tokenizers::Tokenizer,
}

impl QoraTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self.inner.encode(text, false).expect("Failed to encode");
        encoding.get_ids().to_vec()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).expect("Failed to decode")
    }

    /// Build a LLaMA 3 chat prompt.
    /// Format:
    ///   <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    ///   {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
    ///   {message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ///
    pub fn format_chat(&self, user_message: &str, max_tokens: usize) -> Vec<u32> {
        let today = current_date_string();
        let system_content = build_system_prompt(&today, max_tokens);

        let full_text = format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
             {system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
             {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );

        let encoding = self.inner.encode(full_text, false).expect("Failed to encode chat");
        encoding.get_ids().to_vec()
    }
}

fn build_system_prompt(today: &str, max_tokens: usize) -> String {
    let length_hint = if max_tokens <= 100 {
        "IMPORTANT: Keep your response very brief — 1-2 sentences only.\n"
    } else if max_tokens <= 300 {
        "IMPORTANT: Keep your response concise — a few sentences.\n"
    } else if max_tokens <= 500 {
        "Keep your response brief — a short paragraph.\n"
    } else {
        ""
    };

    format!(
        "Cutting Knowledge Date: December 2023\n\
         Today Date: {today}\n\n\
         You are QORA, a helpful AI assistant. \
         You provide accurate, clear responses.\n\
         {length_hint}"
    )
}

/// Get current date as "DD Mon YYYY" string.
fn current_date_string() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = (secs / 86400) as i64;

    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    let month_name = match m {
        1 => "January", 2 => "February", 3 => "March", 4 => "April",
        5 => "May", 6 => "June", 7 => "July", 8 => "August",
        9 => "September", 10 => "October", 11 => "November", 12 => "December",
        _ => "Unknown",
    };

    format!("{d} {month_name} {y}")
}
