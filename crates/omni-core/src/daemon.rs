use anyhow::{Result, Context};
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
    pub config: Config,
    pub brain: Arc<RwLock<TokenAwareBrain>>,
    pub chain: Arc<ChainOrchestrator>,
    pub portal: Arc<Portal>,
    pub ir: Arc<RwLock<IREngine>>,
    pub scheduler: Arc<AdaptiveScheduler>,
    headless: bool,
    supermini: bool,
}

impl OmniDaemon {
    pub async fn new(config: Config, headless: bool, supermini: bool) -> Result<Self> {
        info!("Initializing OMNI-HIVE components with Self-Healing capabilities...");

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

        // Initialize brain
        let brain = Arc::new(RwLock::new(TokenAwareBrain::new(&brain_config)?));
        info!("Brain initialized [OK]");

        // Initialize chain
        let chain = Arc::new(ChainOrchestrator::new(&chain_config)?);
        info!("Chain orchestrator initialized [OK]");

        // Initialize portal
        let portal = Arc::new(Portal::new(&portal_config)?);
        info!("Portal initialized [OK]");

        // Initialize IR engine
        let ir = Arc::new(RwLock::new(IREngine::new(&ir_config)?));
        info!("IR engine initialized [OK]");

        // Initialize scheduler
        let scheduler = Arc::new(AdaptiveScheduler::new(&config.scheduler));
        info!("Adaptive scheduler initialized [OK]");

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

    pub async fn run(self, addr: SocketAddr) -> Result<()> {
        let daemon = Arc::new(self);

        // Setup Panic Hook for Self-Healing
        let daemon_panic = daemon.clone();
        std::panic::set_hook(Box::new(move |info| {
            let msg = info.payload().downcast_ref::<&str>().unwrap_or(&"unknown");
            let loc = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())).unwrap_or_default();
            error!("CRITICAL PANIC at {}: {}. Triggering emergency mutation...", loc, msg);
            
            // In a real panic we can't do much async, but we can signal the watcher
            let _ = std::fs::write("C:\\Users\\Ouroboros\\Desktop\\hive_panic.log", format!("PANIC: {} at {}", msg, loc));
        }));

        // Start API server
        let app = create_router(daemon.clone());
        let listener = tokio::net::TcpListener::bind(addr).await
            .context("Failed to bind API server")?;

        // Start background tasks
        let d1 = daemon.clone();
        tokio::spawn(async move { d1.run_scheduler_loop().await; });

        let d2 = daemon.clone();
        tokio::spawn(async move { d2.run_ir_evolution_loop().await; });

        let d3 = daemon.clone();
        tokio::spawn(async move { d3.run_health_monitor().await; });

        let d4 = daemon.clone();
        tokio::spawn(async move { d4.run_error_resolution_service().await; });

        info!("OMNI-HIVE Hyperdaemon is active and monitoring.");
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Automated Error Detection and Resolution Service
    async fn run_error_resolution_service(&self) {
        info!("Error Resolution Service active. Target: 95% Critical Error Reduction.");
        loop {
            // Scan for unresolved errors in the hive_state and log files
            if let Ok(panic_log) = std::fs::read_to_string("C:\\Users\\Ouroboros\\Desktop\\hive_panic.log") {
                warn!("Active panic detected: {}. Invoking Ouroboros mutation...", panic_log);
                
                let mut ir = self.ir.write().await;
                if let Ok(report) = ir.heal_fault(&panic_log).await {
                    info!("Self-heal mutation complete: {}. Resolution verified.", report.summary);
                    let _ = std::fs::remove_file("C:\\Users\\Ouroboros\\Desktop\\hive_panic.log");
                }
            }

            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }

    async fn run_scheduler_loop(&self) {
        loop {
            let interval = self.scheduler.get_interval().await;
            if self.scheduler.is_active().await {
                match self.sync_all().await {
                    Ok(stats) => { 
                        info!("Sync complete: {} items processed", stats.items_processed);
                        self.scheduler.reset_backoff().await;
                    }
                    Err(e) => { 
                        error!("Sync failure detected: {}. Scheduling auto-retry.", e);
                        self.scheduler.backoff().await;
                    }
                }
            }
            tokio::time::sleep(interval).await;
        }
    }

    async fn run_ir_evolution_loop(&self) {
        loop {
            {
                let mut ir = self.ir.write().await;
                if let Ok(report) = ir.evolve().await {
                    if report.mutations > 0 { info!("IR mutation cycle: {} structural changes applied.", report.mutations); }
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        }
    }

    async fn run_health_monitor(&self) {
        loop {
            let health = self.check_health().await;
            if !health.is_healthy {
                warn!("Autonomic drift detected: {:?}", health.issues);
                self.trigger_healing(&health).await;
            }
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    }

    pub async fn sync_all(&self) -> Result<SyncStats> {
        let mut stats = SyncStats::default();
        let results = self.chain.execute_parallel(vec![
            "sync_gemini", "sync_chatgpt", "sync_claude", "sync_perplexity", "sync_grok",
        ]).await.context("Chain execution failed")?;

        for result in results {
            stats.items_processed += result.items;
            stats.errors += result.errors;
        }
        Ok(stats)
    }

    pub async fn check_health(&self) -> HealthReport {
        let mut report = HealthReport::default();
        {
            let brain = self.brain.read().await;
            if brain.token_usage() > 0.95 { report.issues.push("Brain: Critical Token Pressure (>95%)".into()); }
        }
        if self.chain.bottleneck_detected().await { report.issues.push("Chain: Execution Bottleneck".into()); }
        if !self.portal.is_connected().await { report.issues.push("Portal: Interface Disconnected".into()); }
        
        report.is_healthy = report.issues.is_empty();
        report
    }

    async fn trigger_healing(&self, health: &HealthReport) {
        for issue in &health.issues {
            match issue.as_str() {
                s if s.contains("Token Pressure") => {
                    let mut brain = self.brain.write().await;
                    brain.emergency_flush().await;
                }
                s if s.contains("Bottleneck") => { self.chain.scale_up().await; }
                s if s.contains("Disconnected") => { self.portal.reconnect().await; }
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
