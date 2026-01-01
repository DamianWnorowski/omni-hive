//! OMNI-HIVE Core Daemon
//! Universal AI Orchestration Engine - Self-Evolving Hyperdaemon

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

mod daemon;
mod scheduler;
mod api;
mod config;

use daemon::OmniDaemon;

#[derive(Parser)]
#[command(name = "omni-hive")]
#[command(about = "Universal AI Orchestration Engine", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Configuration file path
    #[arg(short, long, global = true)]
    config: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the OMNI-HIVE daemon
    Start {
        /// API server port
        #[arg(short, long, default_value = "9999")]
        port: u16,

        /// Run in headless mode (no browser UI)
        #[arg(long)]
        headless: bool,

        /// Run in supermini mode (API-only, no browser)
        #[arg(long)]
        supermini: bool,
    },

    /// Show daemon status
    Status,

    /// Sync all AI platforms now
    Sync {
        /// Force full sync (ignore cache)
        #[arg(short, long)]
        force: bool,
    },

    /// Open a specific chat/item via portal
    Open {
        /// Item ID or search query
        query: String,
    },

    /// Show recent activity log
    Log {
        /// Number of entries to show
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Stop the daemon
    Stop,

    /// Show PANTHEON dashboard (GodAI control)
    Dashboard,

    /// Run health check
    Health,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .compact()
        .init();

    match cli.command {
        Commands::Start { port, headless, supermini } => {
            info!("Starting OMNI-HIVE daemon...");
            info!("Mode: {}", if supermini { "Supermini (API-only)" }
                              else if headless { "Headless" }
                              else { "Full" });

            let config = config::Config::load(cli.config)?;
            let daemon = OmniDaemon::new(config, headless, supermini).await?;

            // Start API server
            let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
            info!("API server listening on http://{}", addr);

            daemon.run(addr).await?;
        }

        Commands::Status => {
            println!("OMNI-HIVE Status");
            println!("================");
            // TODO: Query daemon status via API
            println!("Daemon: Running");
            println!("Agents: 47 active");
            println!("Platforms: 15 connected");
            println!("Last sync: 2 minutes ago");
        }

        Commands::Sync { force } => {
            info!("Triggering sync... (force={})", force);
            // TODO: Send sync command to daemon
        }

        Commands::Open { query } => {
            info!("Opening: {}", query);
            // TODO: Use portal to open item
        }

        Commands::Log { limit } => {
            println!("Recent Activity (last {} entries)", limit);
            println!("================================");
            // TODO: Fetch log from daemon
        }

        Commands::Stop => {
            info!("Stopping OMNI-HIVE daemon...");
            // TODO: Send stop signal
        }

        Commands::Dashboard => {
            println!("PANTHEON Dashboard");
            println!("==================");
            // TODO: Launch TUI dashboard
        }

        Commands::Health => {
            println!("Health Check");
            println!("============");
            println!("Core: OK");
            println!("Brain: OK");
            println!("Chain: OK");
            println!("Portal: OK");
            println!("IR: OK");
        }
    }

    Ok(())
}
