//! OMNI-HIVE REST API

use axum::{
    routing::{get, post},
    Router, Json, Extension,
    extract::Path,
    response::IntoResponse,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use uuid::Uuid;

use crate::daemon::OmniDaemon;
use omni_chain::Agent;

/// Create the API router
pub fn create_router(daemon: Arc<OmniDaemon>) -> Router {
    Router::new()
        // Health & Status
        .route("/health", get(health))
        .route("/status", get(status))

        // Portal (Claude context injection)
        .route("/portal/search", post(portal_search))
        .route("/portal/inject", post(portal_inject))
        .route("/portal/chats", get(portal_list_chats))
        .route("/portal/chat/:id", get(portal_get_chat))
        .route("/portal/recent", get(portal_recent))
        .route("/portal/relevant", post(portal_relevant))

        // Chain orchestration
        .route("/chain/execute", post(chain_execute))
        .route("/chain/status", get(chain_status))

        // Brain (token management)
        .route("/brain/usage", get(brain_usage))
        .route("/brain/evict", post(brain_evict))

        // IR (self-writing engine)
        .route("/ir/status", get(ir_status))
        .route("/ir/mutate", post(ir_mutate))
        .route("/ir/requirements", get(ir_requirements))

        // Agents
        .route("/agents", get(list_agents))
        .route("/agents/spawn", post(spawn_agent))
        .route("/agents/evolve", post(evolve_agents))
        .route("/agents/:id", get(get_agent))
        .route("/agents/:id/kill", post(kill_agent))

        // PANTHEON (GodAI control)
        .route("/pantheon/observe", get(pantheon_observe))
        .route("/pantheon/intervene", post(pantheon_intervene))
        .route("/pantheon/predict", get(pantheon_predict))

        // Sync
        .route("/sync", post(trigger_sync))
        .route("/sync/status", get(sync_status))

        .layer(Extension(daemon))
}

// === Health & Status ===

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": 0,  // TODO: track uptime
    }))
}

async fn status(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    let health = daemon.check_health().await;
    Json(serde_json::json!({
        "healthy": health.is_healthy,
        "issues": health.issues,
        "mode": "running",
    }))
}

// === Portal API ===

#[derive(Deserialize)]
struct SearchQuery {
    query: String,
    limit: Option<usize>,
}

async fn portal_search(
    Extension(daemon): Extension<Arc<OmniDaemon>>,
    Json(query): Json<SearchQuery>,
) -> impl IntoResponse {
    // TODO: Implement semantic search
    Json(serde_json::json!({
        "results": [],
        "query": query.query,
    }))
}

async fn portal_inject(
    Extension(daemon): Extension<Arc<OmniDaemon>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    // TODO: Inject context
    Json(serde_json::json!({
        "success": true,
        "tokens_injected": 0,
    }))
}

async fn portal_list_chats(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "chats": [],
        "total": 0,
    }))
}

async fn portal_get_chat(
    Extension(daemon): Extension<Arc<OmniDaemon>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "id": id,
        "content": null,
    }))
}

async fn portal_recent(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "recent": [],
    }))
}

async fn portal_relevant(
    Extension(daemon): Extension<Arc<OmniDaemon>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "relevant": [],
    }))
}

// === Chain API ===

async fn chain_execute(
    Extension(daemon): Extension<Arc<OmniDaemon>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "success": true,
        "chain_id": "chain_001",
    }))
}

async fn chain_status(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "active_chains": 0,
        "completed": 0,
        "failed": 0,
    }))
}

// === Brain API ===

async fn brain_usage(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    let brain = daemon.brain.read().await;
    Json(serde_json::json!({
        "token_count": brain.token_count(),
        "token_limit": brain.token_limit(),
        "usage_percent": brain.token_usage() * 100.0,
        "budget": {
            "system": 5,
            "context": 60,
            "response": 25,
            "reserve": 10,
        }
    }))
}

async fn brain_evict(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    let mut brain = daemon.brain.write().await;
    brain.proactive_evict().await;
    Json(serde_json::json!({
        "success": true,
        "new_usage": brain.token_usage() * 100.0,
    }))
}

// === IR API ===

async fn ir_status(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    let ir = daemon.ir.read().await;
    let nodes = ir.node_count().await;
    let edges = ir.edge_count().await;
    let generation = ir.generation().await;
    Json(serde_json::json!({
        "nodes": nodes,
        "edges": edges,
        "generation": generation,
        "last_mutation": null,
    }))
}

async fn ir_mutate(
    Extension(daemon): Extension<Arc<OmniDaemon>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let mut ir = daemon.ir.write().await;
    // TODO: Apply mutation
    Json(serde_json::json!({
        "success": true,
        "mutations_applied": 0,
    }))
}

async fn ir_requirements(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    let ir = daemon.ir.read().await;
    Json(serde_json::json!({
        "coverage": 0.0,
        "met": 0,
        "unmet": 0,
        "requirements": [],
    }))
}

// === Agents API ===

#[derive(Deserialize)]
struct SpawnRequest {
    name: String,
    #[serde(rename = "type")]
    agent_type: String,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    overloaded: bool,  // For testing MITOSIS
}

async fn list_agents(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    let macro_agent = daemon.chain.macro_agent();
    let agents = macro_agent.list_agents();

    // Group by type
    let mut by_type: HashMap<String, usize> = HashMap::new();
    for agent in &agents {
        *by_type.entry(agent.agent_type.clone()).or_insert(0) += 1;
    }

    Json(serde_json::json!({
        "agents": agents.iter().map(|a| serde_json::json!({
            "id": a.id.to_string(),
            "name": a.name,
            "type": a.agent_type,
            "state": format!("{:?}", a.state),
            "generation": a.generation,
            "parent_id": a.parent_id.map(|id| id.to_string()),
            "children": a.children.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
            "capabilities": a.capabilities,
            "stats": {
                "invocations": a.stats.invocations,
                "successes": a.stats.successes,
                "failures": a.stats.failures,
                "success_rate": a.stats.success_rate(),
                "avg_latency_ms": a.stats.avg_latency_ms,
                "queue_depth": a.stats.queue_depth,
                "is_overloaded": a.stats.is_overloaded(),
            }
        })).collect::<Vec<_>>(),
        "total": agents.len(),
        "by_type": by_type,
    }))
}

async fn spawn_agent(
    Extension(daemon): Extension<Arc<OmniDaemon>>,
    Json(payload): Json<SpawnRequest>,
) -> impl IntoResponse {
    let macro_agent = daemon.chain.macro_agent();

    let mut agent = Agent::new(&payload.name, &payload.agent_type);
    for cap in payload.capabilities {
        agent = agent.with_capability(cap);
    }

    // For testing: simulate overloaded agent that will trigger MITOSIS
    if payload.overloaded {
        agent.stats.queue_depth = 150;
        agent.stats.avg_latency_ms = 1500.0;
        agent.stats.invocations = 100;
        agent.stats.successes = 95;
    }

    let id = macro_agent.spawn(agent);

    Json(serde_json::json!({
        "success": true,
        "agent_id": id.to_string(),
    }))
}

async fn evolve_agents(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    let macro_agent = daemon.chain.macro_agent();
    let report = macro_agent.evolve();

    Json(serde_json::json!({
        "success": true,
        "evolution_report": {
            "splits": report.splits,       // MITOSIS count
            "kills": report.kills,         // APOPTOSIS count
            "merges": report.merges,
            "mutations": report.mutations,
            "total_agents": report.total_agents,
            "duration_ms": report.duration_ms,
        }
    }))
}

async fn get_agent(
    Extension(daemon): Extension<Arc<OmniDaemon>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let macro_agent = daemon.chain.macro_agent();
    let agents = macro_agent.list_agents();

    if let Ok(uuid) = Uuid::parse_str(&id) {
        if let Some(agent) = agents.iter().find(|a| a.id == uuid) {
            return Json(serde_json::json!({
                "id": agent.id.to_string(),
                "name": agent.name,
                "type": agent.agent_type,
                "state": format!("{:?}", agent.state),
                "generation": agent.generation,
                "parent_id": agent.parent_id.map(|id| id.to_string()),
                "children": agent.children.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                "capabilities": agent.capabilities,
                "stats": agent.stats,
                "created_at": agent.created_at.to_rfc3339(),
            }));
        }
    }

    Json(serde_json::json!({
        "error": "Agent not found",
        "id": id,
    }))
}

async fn kill_agent(
    Extension(daemon): Extension<Arc<OmniDaemon>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let macro_agent = daemon.chain.macro_agent();

    if let Ok(uuid) = Uuid::parse_str(&id) {
        if let Some(killed) = macro_agent.kill(&uuid) {
            return Json(serde_json::json!({
                "success": true,
                "agent_id": id,
                "final_state": format!("{:?}", killed.state),
            }));
        }
    }

    Json(serde_json::json!({
        "success": false,
        "error": "Agent not found",
        "agent_id": id,
    }))
}

// === PANTHEON API ===

async fn pantheon_observe(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "observations": {
            "ai_layer": {},
            "system_layer": {},
            "agent_layer": {},
            "data_layer": {},
        }
    }))
}

async fn pantheon_intervene(
    Extension(daemon): Extension<Arc<OmniDaemon>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    Json(serde_json::json!({
        "success": true,
        "intervention": null,
    }))
}

async fn pantheon_predict(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "predictions": {
            "budget_exhaustion": null,
            "rate_limit_risk": 0.0,
            "performance_trajectory": "stable",
            "failure_probability": 0.01,
        }
    }))
}

// === Sync API ===

async fn trigger_sync(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    match daemon.sync_all().await {
        Ok(stats) => Json(serde_json::json!({
            "success": true,
            "items_processed": stats.items_processed,
            "errors": stats.errors,
        })),
        Err(e) => Json(serde_json::json!({
            "success": false,
            "error": e.to_string(),
        })),
    }
}

async fn sync_status(Extension(daemon): Extension<Arc<OmniDaemon>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "last_sync": null,
        "next_sync": null,
        "platforms": [],
    }))
}
