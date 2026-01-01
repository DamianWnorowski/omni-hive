# OMNI-HIVE REST API

Base URL: `http://localhost:7777`

## Health & Status

### GET /health

Returns system health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_secs": 3600
}
```

### GET /status

Returns detailed system status.

**Response:**
```json
{
  "healthy": true,
  "issues": [],
  "mode": "running"
}
```

---

## Portal API

The Portal provides AI context injection and semantic search across all indexed content.

### POST /portal/search

Search across all indexed chats and content.

**Request:**
```json
{
  "query": "code review discussion",
  "limit": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "chat_123",
      "title": "PR Review: Auth Module",
      "score": 0.95,
      "snippet": "...discussed the authentication flow..."
    }
  ],
  "query": "code review discussion"
}
```

### POST /portal/inject

Inject context into the current AI session.

**Request:**
```json
{
  "context_id": "chat_123",
  "tokens_budget": 4000
}
```

**Response:**
```json
{
  "success": true,
  "tokens_injected": 3847
}
```

### GET /portal/chats

List all indexed chats.

**Response:**
```json
{
  "chats": [
    {
      "id": "chat_123",
      "title": "Project Discussion",
      "platform": "claude",
      "last_updated": "2025-01-01T12:00:00Z"
    }
  ],
  "total": 156
}
```

### GET /portal/chat/:id

Get full chat content.

**Response:**
```json
{
  "id": "chat_123",
  "title": "Project Discussion",
  "messages": [...],
  "token_count": 8500
}
```

### GET /portal/recent

Get recently accessed content.

**Response:**
```json
{
  "recent": [
    {
      "id": "chat_456",
      "accessed_at": "2025-01-01T11:30:00Z"
    }
  ]
}
```

### POST /portal/relevant

Get content relevant to the current topic.

**Request:**
```json
{
  "current_context": "Discussing database optimization...",
  "limit": 5
}
```

**Response:**
```json
{
  "relevant": [
    {
      "id": "chat_789",
      "relevance_score": 0.89,
      "title": "PostgreSQL Tuning Session"
    }
  ]
}
```

---

## Chain API

Chain orchestration for complex multi-step operations.

### POST /chain/execute

Execute a chain of operations.

**Request:**
```json
{
  "chain_type": "sequential",
  "steps": [
    {"action": "analyze", "target": "code.rs"},
    {"action": "summarize"},
    {"action": "generate_tests"}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "chain_id": "chain_001",
  "status": "running"
}
```

### GET /chain/status

Get status of all active chains.

**Response:**
```json
{
  "active_chains": 3,
  "completed": 47,
  "failed": 2
}
```

---

## Brain API

Token-aware context management.

### GET /brain/usage

Get current token usage.

**Response:**
```json
{
  "token_count": 45000,
  "token_limit": 128000,
  "usage_percent": 35.2,
  "budget": {
    "system": 5,
    "context": 60,
    "response": 25,
    "reserve": 10
  }
}
```

### POST /brain/evict

Trigger proactive eviction of low-priority content.

**Response:**
```json
{
  "success": true,
  "new_usage": 28.5
}
```

---

## IR API

Self-writing Intermediate Representation engine.

### GET /ir/status

Get IR engine status.

**Response:**
```json
{
  "nodes": 1247,
  "edges": 3891,
  "generation": 42,
  "last_mutation": "2025-01-01T11:45:00Z"
}
```

### POST /ir/mutate

Apply mutations to the IR.

**Request:**
```json
{
  "mutations": [
    {"type": "add_node", "data": {...}},
    {"type": "modify_edge", "from": "n1", "to": "n2", "weight": 0.8}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "mutations_applied": 2
}
```

### GET /ir/requirements

Get requirements coverage status.

**Response:**
```json
{
  "coverage": 0.87,
  "met": 42,
  "unmet": 6,
  "requirements": [
    {"id": "req_001", "status": "met", "description": "..."}
  ]
}
```

---

## Agents API

Macro-agent orchestration.

### GET /agents

List all agents.

**Response:**
```json
{
  "agents": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "analyzer",
      "type": "reasoning",
      "state": "Active",
      "generation": 3,
      "parent_id": null,
      "children": ["..."],
      "capabilities": ["code_analysis", "summarization"],
      "stats": {
        "invocations": 156,
        "successes": 152,
        "failures": 4,
        "success_rate": 0.974,
        "avg_latency_ms": 245.5,
        "queue_depth": 3,
        "is_overloaded": false
      }
    }
  ],
  "total": 47,
  "by_type": {
    "reasoning": 12,
    "extraction": 20,
    "generation": 8,
    "indexing": 7
  }
}
```

### POST /agents/spawn

Spawn a new agent.

**Request:**
```json
{
  "name": "code_reviewer",
  "type": "reasoning",
  "capabilities": ["code_analysis", "security_review"],
  "overloaded": false
}
```

**Response:**
```json
{
  "success": true,
  "agent_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

### POST /agents/evolve

Trigger agent evolution cycle (MITOSIS/APOPTOSIS).

**Response:**
```json
{
  "success": true,
  "evolution_report": {
    "splits": 3,
    "kills": 2,
    "merges": 1,
    "mutations": 5,
    "total_agents": 49,
    "duration_ms": 127
  }
}
```

### GET /agents/:id

Get specific agent details.

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "analyzer",
  "type": "reasoning",
  "state": "Active",
  "generation": 3,
  "capabilities": ["code_analysis"],
  "stats": {...},
  "created_at": "2025-01-01T10:00:00Z"
}
```

### POST /agents/:id/kill

Terminate an agent.

**Response:**
```json
{
  "success": true,
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "final_state": "Dead"
}
```

---

## PANTHEON API

GodAI omniscient control system.

### GET /pantheon/observe

Get current observations across all layers.

**Response:**
```json
{
  "observations": {
    "ai_layer": {
      "active_requests": 12,
      "total_tokens_today": 2400000,
      "error_rate": 0.002
    },
    "system_layer": {
      "cpu_percent": 15.3,
      "memory_mb": 156,
      "disk_io_mb_s": 2.4
    },
    "agent_layer": {
      "total_agents": 49,
      "active": 23,
      "idle": 26
    },
    "data_layer": {
      "db_size_mb": 450,
      "cache_hit_rate": 0.92
    }
  }
}
```

### POST /pantheon/intervene

Trigger a manual intervention.

**Request:**
```json
{
  "intervention_type": "scale_up",
  "target": "reasoning_agents",
  "parameters": {
    "count": 5
  }
}
```

**Response:**
```json
{
  "success": true,
  "intervention": {
    "id": "int_001",
    "type": "scale_up",
    "result": "5 agents spawned"
  }
}
```

### GET /pantheon/predict

Get predictive analysis.

**Response:**
```json
{
  "predictions": {
    "budget_exhaustion": "2025-01-01T18:00:00Z",
    "rate_limit_risk": 0.15,
    "performance_trajectory": "stable",
    "failure_probability": 0.01,
    "recommendations": [
      "Consider switching to Groq for fast queries to reduce GPT-4 load"
    ]
  }
}
```

---

## Sync API

Platform synchronization.

### POST /sync

Trigger immediate sync of all platforms.

**Response:**
```json
{
  "success": true,
  "items_processed": 47,
  "errors": 0
}
```

### GET /sync/status

Get sync status.

**Response:**
```json
{
  "last_sync": "2025-01-01T11:30:00Z",
  "next_sync": "2025-01-01T12:00:00Z",
  "platforms": [
    {"name": "gemini", "status": "synced", "items": 23},
    {"name": "claude", "status": "synced", "items": 18},
    {"name": "chatgpt", "status": "pending", "items": 0}
  ]
}
```

---

## Error Responses

All endpoints may return error responses:

```json
{
  "error": "Agent not found",
  "code": "AGENT_NOT_FOUND",
  "details": {
    "agent_id": "invalid-id"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request body |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `TOKEN_BUDGET_EXCEEDED` | 507 | Token budget exhausted |
