use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct BitNetConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_bos")]
    pub bos_token_id: usize,
    #[serde(default = "default_eos")]
    pub eos_token_id: usize,
}

fn default_eps() -> f64 { 1e-5 }
fn default_theta() -> f64 { 500000.0 }
fn default_max_pos() -> usize { 4096 }
fn default_bos() -> usize { 128000 }
fn default_eos() -> usize { 128001 }

impl BitNetConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
