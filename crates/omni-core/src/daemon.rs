//! OMNI-HIVE Daemon - Core orchestration loop

use anyhow::Result;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

use crate::config::Config;
use crate::scheduler::AdaptiveScheduler;
use crate::api::create_router;

// Re-export library types for internal use
use omni_brain::{TokenAwareBrain, BrainConfig as LibBrainConfig, TokenBudget};
use omni_chain::{ChainOrchestrator, ChainConfig as LibChainConfig, ChainResult};
use omni_portal::{Portal, PortalConfig as LibPortalConfig};
use omni_ir::{IREngine, IRConfig as LibIRConfig};

/// The main OMNI-HIVE daemon
pub struct OmniDaemon {
    /// Configuration
    pub config: Config,

    /// Token-aware context management
    pub brain: Arc<RwLock<TokenAwareBrain>>,

    /// Chain orchestration engine
    pub chain: Arc<ChainOrchestrator>,

    /// AI re-entry portal
    pub portal: Arc<Portal>,

    /// Self-writing IR engine
    pub ir: Arc<RwLock<IREngine>>,

    /// Adaptive scheduler
    pub scheduler: Arc<AdaptiveScheduler>,

    /// Running mode
    headless: bool,
    supermini: bool,
}

impl OmniDaemon {
    pub async fn new(config: Config, headless: bool, supermini: bool) -> Result<Self> {
        info!("Initializing OMNI-HIVE components...");

        // Convert config types
        let brain_config = LibBrainConfig {
            budget: TokenBudget {
                system_pct: config.brain.system_pct,
                context_pct: config.brain.context_pct,
                response_pct: config.brain.response_pct,
                reserve_pct: config.brain.reserve_pct,
            },
            max_tokens: config.brain.max_tokens,
            stream_threshold: config.brain.stream_threshold,
            eviction_batch_size: config.brain.eviction_batch_size,
        };

        let chain_config = LibChainConfig {
            max_parallel: config.chain.max_parallel,
            timeout_secs: config.chain.timeout_secs,
            enable_infinite: config.chain.enable_infinite,
        };

        let portal_config = LibPortalConfig {
            db_path: config.portal.db_path.clone(),
            enable_semantic: config.portal.enable_semantic,
            embedding_model: config.portal.embedding_model.clone(),
        };

        let ir_config = LibIRConfig {
            max_heal_iterations: config.ir.max_heal_iterations,
            mutation_threshold: config.ir.mutation_threshold,
            enable_self_write: config.ir.enable_self_write,
        };

        // Initialize brain (token-aware context)
        let brain = Arc::new(RwLock::new(TokenAwareBrain::new(&brain_config)?));
        info!("Brain initialized");

        // Initialize chain orchestrator
        let chain = Arc::new(ChainOrchestrator::new(&chain_config)?);
        info!("Chain orchestrator initialized");

        // Initialize portal
        let portal = Arc::new(Portal::new(&portal_config)?);
        info!("Portal initialized");

        // Initialize IR engine
        let ir = Arc::new(RwLock::new(IREngine::new(&ir_config)?));
        info!("IR engine initialized");

        // Initialize scheduler
        let scheduler = Arc::new(AdaptiveScheduler::new(&config.scheduler));
        info!("Adaptive scheduler initialized");

        Ok(Self {
            config,
            brain,
            chain,
            portal,
            ir,
            scheduler,
            headless,
            supermini,
        })
    }

    /// Run the daemon
    pub async fn run(self, addr: SocketAddr) -> Result<()> {
        let daemon = Arc::new(self);

        // Start API server
        let app = create_router(daemon.clone());
        let listener = tokio::net::TcpListener::bind(addr).await?;

        // Start background tasks
        let daemon_clone = daemon.clone();
        tokio::spawn(async move {
            daemon_clone.run_scheduler_loop().await;
        });

        let daemon_clone = daemon.clone();
        tokio::spawn(async move {
            daemon_clone.run_ir_evolution_loop().await;
        });

        let daemon_clone = daemon.clone();
        tokio::spawn(async move {
            daemon_clone.run_health_monitor().await;
        });

        // Run API server
        info!("OMNI-HIVE daemon is running");
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Adaptive scheduler loop
    async fn run_scheduler_loop(&self) {
        loop {
            let interval = self.scheduler.get_interval().await;

            if self.scheduler.is_active().await {
                match self.sync_all().await {
                    Ok(stats) => {
                        info!("Sync complete: {} items processed", stats.items_processed);
                    }
                    Err(e) => {
                        error!("Sync failed: {}", e);
                    }
                }
            }

            tokio::time::sleep(interval).await;
        }
    }

    /// IR self-evolution loop
    async fn run_ir_evolution_loop(&self) {
        loop {
            {
                let mut ir = self.ir.write().await;
                match ir.evolve().await {
                    Ok(report) => {
                        if report.mutations > 0 {
                            info!("IR evolved: {} mutations", report.mutations);
                        }
                    }
                    Err(e) => {
                        warn!("IR evolution failed: {}", e);
                    }
                }
            }

            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        }
    }

    /// Health monitoring loop
    async fn run_health_monitor(&self) {
        loop {
            let health = self.check_health().await;

            if !health.is_healthy {
                warn!("Health check failed: {:?}", health.issues);
                // Trigger self-healing
                self.trigger_healing(&health).await;
            }

            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }

    /// Sync all platforms
    pub async fn sync_all(&self) -> Result<SyncStats> {
        let mut stats = SyncStats::default();

        // Use chain orchestrator for parallel sync
        let results = self.chain.execute_parallel(vec![
            "sync_gemini",
            "sync_chatgpt",
            "sync_claude",
            "sync_perplexity",
            "sync_grok",
        ]).await?;

        for result in results {
            stats.items_processed += result.items;
            stats.errors += result.errors;
        }

        Ok(stats)
    }

    /// Check system health
    pub async fn check_health(&self) -> HealthReport {
        let mut report = HealthReport::default();

        // Check brain
        {
            let brain = self.brain.read().await;
            if brain.token_usage() > 0.9 {
                report.issues.push("Brain token usage > 90%".into());
            }
        }

        // Check chain
        if self.chain.bottleneck_detected().await {
            report.issues.push("Chain bottleneck detected".into());
        }

        // Check portal
        if !self.portal.is_connected().await {
            report.issues.push("Portal disconnected".into());
        }

        report.is_healthy = report.issues.is_empty();
        report
    }

    /// Trigger self-healing
    async fn trigger_healing(&self, health: &HealthReport) {
        for issue in &health.issues {
            match issue.as_str() {
                s if s.contains("token usage") => {
                    let mut brain = self.brain.write().await;
                    brain.proactive_evict().await;
                }
                s if s.contains("bottleneck") => {
                    self.chain.scale_up().await;
                }
                s if s.contains("disconnected") => {
                    self.portal.reconnect().await;
                }
                _ => {}
            }
        }
    }
}

#[derive(Default)]
pub struct SyncStats {
    pub items_processed: usize,
    pub errors: usize,
}

#[derive(Default)]
pub struct HealthReport {
    pub is_healthy: bool,
    pub issues: Vec<String>,
}
