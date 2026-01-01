//! OMNI-IR: Self-Writing IR Engine
//!
//! Features:
//! - Global analyzer registry with decorator pattern
//! - AutoAuditHealer with heuristic patches
//! - Requirements coverage analyzer
//! - Dynamic IR mutation engine

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use inventory;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

// ============================================================================
// GLOBAL ANALYZER REGISTRY (Decorator Pattern)
// ============================================================================

/// Trait for IR analyzers - use #[inventory::submit!] to register
#[async_trait]
pub trait Analyzer: Send + Sync {
    /// Unique analyzer name
    fn name(&self) -> &str;

    /// Analyze IR and return findings
    async fn analyze(&self, ir: &IRGraph) -> Result<Vec<Finding>>;

    /// Priority (higher = runs first)
    fn priority(&self) -> u32 { 100 }
}

/// Finding from an analyzer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Finding {
    pub id: Uuid,
    pub analyzer: String,
    pub severity: Severity,
    pub message: String,
    pub location: Option<IRNodeId>,
    pub suggestion: Option<String>,
    pub auto_fixable: bool,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Global registry for analyzers
pub struct AnalyzerRegistry {
    analyzers: DashMap<String, Arc<dyn Analyzer>>,
}

impl AnalyzerRegistry {
    pub fn new() -> Self {
        Self {
            analyzers: DashMap::new(),
        }
    }

    /// Register an analyzer
    pub fn register(&self, analyzer: Arc<dyn Analyzer>) {
        self.analyzers.insert(analyzer.name().to_string(), analyzer);
    }

    /// Get all analyzers sorted by priority
    pub fn get_all(&self) -> Vec<Arc<dyn Analyzer>> {
        let mut analyzers: Vec<_> = self.analyzers.iter().map(|e| e.value().clone()).collect();
        analyzers.sort_by(|a, b| b.priority().cmp(&a.priority()));
        analyzers
    }

    /// Run all analyzers on IR
    pub async fn analyze_all(&self, ir: &IRGraph) -> Vec<Finding> {
        let mut findings = Vec::new();

        for analyzer in self.get_all() {
            if let Ok(results) = analyzer.analyze(ir).await {
                findings.extend(results);
            }
        }

        findings.sort_by(|a, b| b.severity.cmp(&a.severity));
        findings
    }
}

impl Default for AnalyzerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// IR GRAPH
// ============================================================================

pub type IRNodeId = Uuid;
pub type IREdgeId = Uuid;

/// IR Node types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IRNodeType {
    Function { name: String, params: Vec<String> },
    Variable { name: String, mutable: bool },
    Constant { value: serde_json::Value },
    Operation { op: String },
    Control { kind: ControlKind },
    Custom(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ControlKind {
    If,
    Loop,
    Match,
    Return,
    Break,
    Continue,
}

/// IR Node
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IRNode {
    pub id: IRNodeId,
    pub node_type: IRNodeType,
    pub inputs: Vec<IRNodeId>,
    pub outputs: Vec<IRNodeId>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub generation: u64,
}

impl IRNode {
    pub fn new(node_type: IRNodeType) -> Self {
        Self {
            id: Uuid::new_v4(),
            node_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            generation: 0,
        }
    }
}

/// IR Edge
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IREdge {
    pub id: IREdgeId,
    pub from: IRNodeId,
    pub to: IRNodeId,
    pub edge_type: String,
    pub weight: f32,
}

/// IR Graph representation
#[derive(Clone, Debug, Default)]
pub struct IRGraph {
    nodes: HashMap<IRNodeId, IRNode>,
    edges: HashMap<IREdgeId, IREdge>,
    generation: u64,
}

impl IRGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, mut node: IRNode) -> IRNodeId {
        node.generation = self.generation;
        let id = node.id;
        self.nodes.insert(id, node);
        id
    }

    pub fn add_edge(&mut self, from: IRNodeId, to: IRNodeId, edge_type: &str) -> IREdgeId {
        let edge = IREdge {
            id: Uuid::new_v4(),
            from,
            to,
            edge_type: edge_type.to_string(),
            weight: 1.0,
        };
        let id = edge.id;
        self.edges.insert(id, edge);

        // Update node connections
        if let Some(node) = self.nodes.get_mut(&from) {
            node.outputs.push(to);
        }
        if let Some(node) = self.nodes.get_mut(&to) {
            node.inputs.push(from);
        }

        id
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn get_node(&self, id: &IRNodeId) -> Option<&IRNode> {
        self.nodes.get(id)
    }

    pub fn get_nodes(&self) -> impl Iterator<Item = &IRNode> {
        self.nodes.values()
    }

    pub fn increment_generation(&mut self) {
        self.generation += 1;
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

// ============================================================================
// AUTO-AUDIT HEALER
// ============================================================================

/// Heuristic patch for auto-healing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeuristicPatch {
    pub id: Uuid,
    pub finding_id: Uuid,
    pub description: String,
    pub confidence: f32,
    pub mutations: Vec<Mutation>,
}

/// IR Mutation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Mutation {
    AddNode(IRNode),
    RemoveNode(IRNodeId),
    UpdateNode { id: IRNodeId, updates: HashMap<String, serde_json::Value> },
    AddEdge { from: IRNodeId, to: IRNodeId, edge_type: String },
    RemoveEdge(IREdgeId),
    ReplaceSubgraph { old_root: IRNodeId, new_subgraph: Vec<IRNode> },
}

/// AutoAuditHealer - Self-auditing with heuristic patches
pub struct AutoAuditHealer {
    min_confidence: f32,
    max_iterations: usize,
    patches_applied: AtomicU64,
}

impl AutoAuditHealer {
    pub fn new(min_confidence: f32) -> Self {
        Self {
            min_confidence,
            max_iterations: 100,
            patches_applied: AtomicU64::new(0),
        }
    }

    /// Generate patches for findings
    pub fn generate_patches(&self, findings: &[Finding]) -> Vec<HeuristicPatch> {
        findings
            .iter()
            .filter(|f| f.auto_fixable && f.confidence >= self.min_confidence)
            .map(|f| {
                HeuristicPatch {
                    id: Uuid::new_v4(),
                    finding_id: f.id,
                    description: f.suggestion.clone().unwrap_or_default(),
                    confidence: f.confidence,
                    mutations: Vec::new(), // TODO: Generate actual mutations
                }
            })
            .collect()
    }

    /// Apply patches to IR
    pub fn apply_patches(&self, ir: &mut IRGraph, patches: &[HeuristicPatch]) -> usize {
        let mut applied = 0;

        for patch in patches {
            if patch.confidence >= self.min_confidence {
                for mutation in &patch.mutations {
                    match mutation {
                        Mutation::AddNode(node) => {
                            ir.add_node(node.clone());
                            applied += 1;
                        }
                        Mutation::RemoveNode(id) => {
                            ir.nodes.remove(id);
                            applied += 1;
                        }
                        Mutation::AddEdge { from, to, edge_type } => {
                            ir.add_edge(*from, *to, edge_type);
                            applied += 1;
                        }
                        _ => {}
                    }
                }
            }
        }

        self.patches_applied.fetch_add(applied as u64, Ordering::Relaxed);
        applied
    }

    pub fn total_patches_applied(&self) -> u64 {
        self.patches_applied.load(Ordering::Relaxed)
    }
}

// ============================================================================
// REQUIREMENTS COVERAGE
// ============================================================================

/// A requirement to be covered
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Requirement {
    pub id: String,
    pub description: String,
    pub category: String,
    pub covered: bool,
    pub coverage_nodes: Vec<IRNodeId>,
}

/// Requirements coverage analyzer
pub struct RequirementsCoverage {
    requirements: DashMap<String, Requirement>,
}

impl RequirementsCoverage {
    pub fn new() -> Self {
        Self {
            requirements: DashMap::new(),
        }
    }

    pub fn add_requirement(&self, req: Requirement) {
        self.requirements.insert(req.id.clone(), req);
    }

    pub fn mark_covered(&self, id: &str, nodes: Vec<IRNodeId>) {
        if let Some(mut req) = self.requirements.get_mut(id) {
            req.covered = true;
            req.coverage_nodes = nodes;
        }
    }

    pub fn coverage_percent(&self) -> f32 {
        let total = self.requirements.len();
        if total == 0 {
            return 100.0;
        }

        let covered = self.requirements.iter().filter(|r| r.covered).count();
        covered as f32 / total as f32 * 100.0
    }

    pub fn unmet_requirements(&self) -> Vec<Requirement> {
        self.requirements
            .iter()
            .filter(|r| !r.covered)
            .map(|r| r.clone())
            .collect()
    }

    pub fn stats(&self) -> RequirementStats {
        let total = self.requirements.len();
        let covered = self.requirements.iter().filter(|r| r.covered).count();

        RequirementStats {
            total,
            covered,
            uncovered: total - covered,
            coverage_percent: if total > 0 { covered as f32 / total as f32 * 100.0 } else { 100.0 },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RequirementStats {
    pub total: usize,
    pub covered: usize,
    pub uncovered: usize,
    pub coverage_percent: f32,
}

// ============================================================================
// DYNAMIC IR MUTATION ENGINE
// ============================================================================

/// Mutation strategy
#[derive(Clone, Debug)]
pub enum MutationStrategy {
    Random,
    Guided { target_coverage: f32 },
    Evolutionary { fitness_fn: String },
}

/// Dynamic IR mutation engine
pub struct MutationEngine {
    strategy: MutationStrategy,
    mutation_rate: f32,
    generation: AtomicU64,
}

impl MutationEngine {
    pub fn new(strategy: MutationStrategy) -> Self {
        Self {
            strategy,
            mutation_rate: 0.1,
            generation: AtomicU64::new(0),
        }
    }

    /// Generate mutations for IR
    pub fn generate_mutations(&self, ir: &IRGraph) -> Vec<Mutation> {
        let mut mutations = Vec::new();

        match &self.strategy {
            MutationStrategy::Random => {
                // Random mutation: add a new node
                if rand::random::<f32>() < self.mutation_rate {
                    let node = IRNode::new(IRNodeType::Custom("mutated".into()));
                    mutations.push(Mutation::AddNode(node));
                }
            }
            MutationStrategy::Guided { target_coverage } => {
                // TODO: Guided mutation based on coverage
            }
            MutationStrategy::Evolutionary { fitness_fn } => {
                // TODO: Evolutionary mutation based on fitness
            }
        }

        mutations
    }

    pub fn current_generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    pub fn increment_generation(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }
}

// ============================================================================
// IR ENGINE (Main Interface)
// ============================================================================

/// Configuration for IR engine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IRConfig {
    pub max_heal_iterations: usize,
    pub mutation_threshold: f32,
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

/// Evolution report from IR engine
#[derive(Debug, Serialize)]
pub struct EvolutionReport {
    pub mutations: usize,
    pub patches_applied: usize,
    pub findings_resolved: usize,
    pub generation: u64,
}

/// Main IR Engine
pub struct IREngine {
    config: IRConfig,
    graph: RwLock<IRGraph>,
    registry: Arc<AnalyzerRegistry>,
    healer: Arc<AutoAuditHealer>,
    requirements: Arc<RequirementsCoverage>,
    mutation_engine: Arc<MutationEngine>,
}

impl IREngine {
    pub fn new(config: &IRConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            graph: RwLock::new(IRGraph::new()),
            registry: Arc::new(AnalyzerRegistry::new()),
            healer: Arc::new(AutoAuditHealer::new(config.mutation_threshold)),
            requirements: Arc::new(RequirementsCoverage::new()),
            mutation_engine: Arc::new(MutationEngine::new(MutationStrategy::Random)),
        })
    }

    pub async fn node_count(&self) -> usize {
        self.graph.read().await.node_count()
    }

    pub async fn edge_count(&self) -> usize {
        self.graph.read().await.edge_count()
    }

    pub async fn generation(&self) -> u64 {
        self.graph.read().await.generation()
    }

    /// Register an analyzer
    pub fn register_analyzer(&self, analyzer: Arc<dyn Analyzer>) {
        self.registry.register(analyzer);
    }

    /// Run evolution cycle
    pub async fn evolve(&mut self) -> Result<EvolutionReport> {
        let mut graph = self.graph.write().await;

        // 1. Analyze IR
        let findings = self.registry.analyze_all(&graph).await;

        // 2. Generate patches
        let patches = self.healer.generate_patches(&findings);

        // 3. Apply patches
        let patches_applied = self.healer.apply_patches(&mut graph, &patches);

        // 4. Generate mutations
        let mutations = self.mutation_engine.generate_mutations(&graph);
        let mutation_count = mutations.len();

        // 5. Apply mutations
        for mutation in mutations {
            match mutation {
                Mutation::AddNode(node) => { graph.add_node(node); }
                _ => {}
            }
        }

        // 6. Increment generation
        graph.increment_generation();
        self.mutation_engine.increment_generation();

        Ok(EvolutionReport {
            mutations: mutation_count,
            patches_applied,
            findings_resolved: patches_applied,
            generation: graph.generation(),
        })
    }

    /// Get requirements coverage
    pub fn requirements(&self) -> &RequirementsCoverage {
        &self.requirements
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_graph() {
        let mut graph = IRGraph::new();
        let node = IRNode::new(IRNodeType::Function {
            name: "test".into(),
            params: vec![]
        });

        let id = graph.add_node(node);
        assert_eq!(graph.node_count(), 1);
        assert!(graph.get_node(&id).is_some());
    }

    #[test]
    fn test_analyzer_registry() {
        let registry = AnalyzerRegistry::new();
        assert_eq!(registry.get_all().len(), 0);
    }
}
