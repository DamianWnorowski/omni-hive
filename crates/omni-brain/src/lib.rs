//! OMNI-BRAIN: Token-Aware Context Management
//!
//! Budget allocation: 5% system, 60% context, 25% response, 10% reserve

use anyhow::Result;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use uuid::Uuid;

/// Token budget allocation percentages
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenBudget {
    /// System prompt percentage (default 5%)
    pub system_pct: f32,
    /// Context percentage (default 60%)
    pub context_pct: f32,
    /// Response percentage (default 25%)
    pub response_pct: f32,
    /// Reserve percentage (default 10%)
    pub reserve_pct: f32,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            system_pct: 0.05,
            context_pct: 0.60,
            response_pct: 0.25,
            reserve_pct: 0.10,
        }
    }
}

/// Individual context entry with importance scoring
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextEntry {
    pub id: Uuid,
    pub content: String,
    pub token_count: usize,
    pub importance: f32,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u64,
    pub source: ContextSource,
    pub tags: Vec<String>,
}

impl ContextEntry {
    pub fn new(content: String, token_count: usize, source: ContextSource) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content,
            token_count,
            importance: 0.5,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            source,
            tags: Vec::new(),
        }
    }

    /// Compute eviction priority (lower = evict first)
    pub fn eviction_priority(&self) -> f32 {
        let age_hours = (Utc::now() - self.last_accessed).num_hours() as f32;
        let recency_factor = 1.0 / (1.0 + age_hours * 0.1);
        let frequency_factor = (self.access_count as f32).ln_1p() / 10.0;

        self.importance * 0.5 + recency_factor * 0.3 + frequency_factor * 0.2
    }
}

impl Eq for ContextEntry {}

impl PartialEq for ContextEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Ord for ContextEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.eviction_priority()
            .partial_cmp(&other.eviction_priority())
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ContextEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContextSource {
    User,
    System,
    Retrieved,
    Generated,
    External(String),
}

/// Configuration for TokenAwareBrain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainConfig {
    pub budget: TokenBudget,
    pub max_tokens: usize,
    pub stream_threshold: f32,
    pub eviction_batch_size: usize,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            budget: TokenBudget::default(),
            max_tokens: 128000, // 128k default
            stream_threshold: 0.50,
            eviction_batch_size: 10,
        }
    }
}

/// Token-aware context management brain
pub struct TokenAwareBrain {
    config: BrainConfig,
    entries: DashMap<Uuid, ContextEntry>,
    total_tokens: AtomicU64,
    generation: AtomicU64,
}

impl TokenAwareBrain {
    pub fn new(config: &BrainConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            entries: DashMap::new(),
            total_tokens: AtomicU64::new(0),
            generation: AtomicU64::new(0),
        })
    }

    /// Get current token count
    pub fn token_count(&self) -> usize {
        self.total_tokens.load(AtomicOrdering::Relaxed) as usize
    }

    /// Get token limit
    pub fn token_limit(&self) -> usize {
        self.config.max_tokens
    }

    /// Get current usage as percentage (0.0 - 1.0)
    pub fn token_usage(&self) -> f32 {
        self.token_count() as f32 / self.token_limit() as f32
    }

    /// Get available context budget in tokens
    pub fn context_budget(&self) -> usize {
        (self.config.max_tokens as f32 * self.config.budget.context_pct) as usize
    }

    /// Get available response budget in tokens
    pub fn response_budget(&self) -> usize {
        (self.config.max_tokens as f32 * self.config.budget.response_pct) as usize
    }

    /// Add a context entry
    pub fn add_entry(&self, entry: ContextEntry) -> Result<Uuid> {
        let id = entry.id;
        let tokens = entry.token_count;

        self.entries.insert(id, entry);
        self.total_tokens.fetch_add(tokens as u64, AtomicOrdering::Relaxed);
        self.generation.fetch_add(1, AtomicOrdering::Relaxed);

        // Check if we need to evict
        if self.token_usage() > 0.9 {
            self.evict_low_priority(self.config.eviction_batch_size);
        }

        Ok(id)
    }

    /// Get an entry by ID (updates access stats)
    pub fn get_entry(&self, id: &Uuid) -> Option<ContextEntry> {
        self.entries.get_mut(id).map(|mut entry| {
            entry.last_accessed = Utc::now();
            entry.access_count += 1;
            entry.clone()
        })
    }

    /// Remove an entry
    pub fn remove_entry(&self, id: &Uuid) -> Option<ContextEntry> {
        self.entries.remove(id).map(|(_, entry)| {
            self.total_tokens.fetch_sub(entry.token_count as u64, AtomicOrdering::Relaxed);
            entry
        })
    }

    /// Evict lowest priority entries
    fn evict_low_priority(&self, count: usize) -> Vec<Uuid> {
        let mut heap: BinaryHeap<(i64, Uuid)> = self.entries
            .iter()
            .map(|e| {
                let priority = (e.eviction_priority() * 1000.0) as i64;
                (-priority, e.id) // Negate for min-heap behavior
            })
            .collect();

        let mut evicted = Vec::new();
        for _ in 0..count {
            if let Some((_, id)) = heap.pop() {
                if self.remove_entry(&id).is_some() {
                    evicted.push(id);
                }
            }
        }
        evicted
    }

    /// Proactive eviction to free up space
    pub async fn proactive_evict(&mut self) {
        let target_usage = 0.7; // Evict down to 70%
        let current = self.token_usage();

        if current > target_usage {
            let tokens_to_free = ((current - target_usage) * self.token_limit() as f32) as usize;
            let avg_entry_size = self.token_count() / self.entries.len().max(1);
            let entries_to_evict = (tokens_to_free / avg_entry_size.max(1)).max(1);

            self.evict_low_priority(entries_to_evict);
        }
    }

    /// Get all entries sorted by importance
    pub fn get_top_entries(&self, limit: usize) -> Vec<ContextEntry> {
        let mut entries: Vec<_> = self.entries.iter().map(|e| e.clone()).collect();
        entries.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(Ordering::Equal));
        entries.truncate(limit);
        entries
    }

    /// Search entries by tag
    pub fn search_by_tag(&self, tag: &str) -> Vec<ContextEntry> {
        self.entries
            .iter()
            .filter(|e| e.tags.contains(&tag.to_string()))
            .map(|e| e.clone())
            .collect()
    }

    /// Get current generation (increments on every change)
    pub fn generation(&self) -> u64 {
        self.generation.load(AtomicOrdering::Relaxed)
    }

    /// Check if streaming should be enabled based on usage
    pub fn should_stream(&self) -> bool {
        self.token_usage() > self.config.stream_threshold
    }

    /// Get statistics
    pub fn stats(&self) -> BrainStats {
        BrainStats {
            total_entries: self.entries.len(),
            total_tokens: self.token_count(),
            max_tokens: self.token_limit(),
            usage_percent: self.token_usage() * 100.0,
            generation: self.generation(),
            context_budget: self.context_budget(),
            response_budget: self.response_budget(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct BrainStats {
    pub total_entries: usize,
    pub total_tokens: usize,
    pub max_tokens: usize,
    pub usage_percent: f32,
    pub generation: u64,
    pub context_budget: usize,
    pub response_budget: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_budget_default() {
        let budget = TokenBudget::default();
        assert!((budget.system_pct + budget.context_pct + budget.response_pct + budget.reserve_pct - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_brain_creation() {
        let config = BrainConfig::default();
        let brain = TokenAwareBrain::new(&config).unwrap();
        assert_eq!(brain.token_count(), 0);
        assert_eq!(brain.token_limit(), 128000);
    }
}
