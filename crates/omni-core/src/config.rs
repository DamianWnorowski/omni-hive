//! OMNI-HIVE Configuration

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Serialize, Deserialize)]
pub struct Config {
    pub brain: BrainConfig,
    pub chain: ChainConfig,
    pub portal: PortalConfig,
    pub ir: IRConfig,
    pub scheduler: SchedulerConfig,
    pub platforms: PlatformsConfig,
}

impl Config {
    pub fn load(path: Option<String>) -> Result<Self> {
        if let Some(path) = path {
            let content = std::fs::read_to_string(path)?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            brain: BrainConfig::default(),
            chain: ChainConfig::default(),
            portal: PortalConfig::default(),
            ir: IRConfig::default(),
            scheduler: SchedulerConfig::default(),
            platforms: PlatformsConfig::default(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BrainConfig {
    /// System prompt percentage (default 5%)
    pub system_pct: f32,
    /// Context percentage (default 60%)
    pub context_pct: f32,
    /// Response percentage (default 25%)
    pub response_pct: f32,
    /// Reserve percentage (default 10%)
    pub reserve_pct: f32,
    /// Stream threshold (start streaming at 50%)
    pub stream_threshold: f32,
    /// Maximum tokens
    pub max_tokens: usize,
    /// Eviction batch size
    pub eviction_batch_size: usize,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            system_pct: 0.05,
            context_pct: 0.60,
            response_pct: 0.25,
            reserve_pct: 0.10,
            stream_threshold: 0.50,
            max_tokens: 128000,
            eviction_batch_size: 10,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ChainConfig {
    /// Maximum parallel chains
    pub max_parallel: usize,
    /// Chain timeout
    pub timeout_secs: u64,
    /// Enable infinite chains
    pub enable_infinite: bool,
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            max_parallel: 10,
            timeout_secs: 300,
            enable_infinite: true,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PortalConfig {
    /// Database path
    pub db_path: PathBuf,
    /// Enable semantic search
    pub enable_semantic: bool,
    /// Embedding model
    pub embedding_model: String,
}

impl Default for PortalConfig {
    fn default() -> Self {
        Self {
            db_path: dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("omni-hive")
                .join("portal.db"),
            enable_semantic: true,
            embedding_model: "all-MiniLM-L6-v2".into(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct IRConfig {
    /// Maximum iterations for healing
    pub max_heal_iterations: usize,
    /// Mutation confidence threshold
    pub mutation_threshold: f32,
    /// Enable self-writing
    pub enable_self_write: bool,
}

impl Default for IRConfig {
    fn default() -> Self {
        Self {
            max_heal_iterations: 100,
            mutation_threshold: 0.85,
            enable_self_write: true,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub idle_interval_ms: u64,
    pub active_interval_ms: u64,
    pub burst_interval_ms: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            idle_interval_ms: 3600000,  // 1 hour
            active_interval_ms: 30000,   // 30 seconds
            burst_interval_ms: 5000,     // 5 seconds
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PlatformsConfig {
    /// Enabled platforms
    pub enabled: Vec<String>,
    /// Platform-specific settings
    pub settings: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for PlatformsConfig {
    fn default() -> Self {
        Self {
            enabled: vec![
                "gemini".into(),
                "chatgpt".into(),
                "claude".into(),
                "perplexity".into(),
                "grok".into(),
            ],
            settings: std::collections::HashMap::new(),
        }
    }
}
