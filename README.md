# OMNI-HIVE

**Universal AI Orchestration Engine - Self-Evolving Rust Hyperdaemon**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

OMNI-HIVE is a high-performance, self-evolving AI orchestration daemon written in Rust. It provides unified access to 100+ AI platforms (web, desktop, API, local) through a single interface with intelligent routing, failover, and context management.

## Features

### Core Capabilities

- **Universal AI Access**: Connect to 15+ web AI platforms, 6+ desktop apps, 9+ API providers, and 5+ local runtimes
- **Token-Aware Brain**: Never hit token limits - proactive streaming windows with zero compaction
- **Hyperbeast 3D Matrix**: N-dimensional chain orchestration with wormhole traversal
- **Macro-Agent Layer**: Self-spawning agents with MITOSIS/APOPTOSIS lifecycle
- **Self-Writing IR Engine**: Auto-healing code with heuristic patches
- **PANTHEON GodAI**: Omniscient monitoring and intervention system

### Supported Platforms

| Category | Platforms |
|----------|-----------|
| **Web AI** | Gemini, ChatGPT, Claude, Copilot, Perplexity, Grok, Mistral, DeepSeek, Phind, You.com, Pi, Poe |
| **Desktop** | ChatGPT Desktop, Claude Desktop, GitHub Copilot, Cursor, Cody |
| **API** | OpenAI, Anthropic, Google, Mistral, Groq, Together, Fireworks, Replicate, HuggingFace |
| **Local** | Ollama, LM Studio, LocalAI, vLLM, text-generation-webui |
| **Multimodal** | DALL-E, Midjourney, Stable Diffusion, Runway, ElevenLabs, Suno |

## Installation

### Prerequisites

- Rust 1.75+ with Cargo
- Chrome/Chromium (for web automation)
- Windows 10/11, macOS, or Linux

### Build from Source

```bash
git clone https://github.com/ouroboros/omni-hive.git
cd omni-hive
cargo build --release
```

The binary will be at `target/release/omni-hive` (or `omni-hive.exe` on Windows).

## Usage

### CLI Commands

```bash
# Start the daemon
omni-hive start

# Start in headless mode (no browser UI)
omni-hive start --headless

# Start in supermini mode (API-only, minimal resources)
omni-hive start --supermini

# Check status
omni-hive status

# Force sync all platforms
omni-hive sync

# Run health check
omni-hive health

# Show PANTHEON dashboard
omni-hive dashboard

# Stop the daemon
omni-hive stop
```

### Configuration

Create a config file at `~/.config/omni-hive/config.json`:

```json
{
  "brain": {
    "max_tokens": 128000,
    "system_pct": 0.05,
    "context_pct": 0.60,
    "response_pct": 0.25,
    "reserve_pct": 0.10
  },
  "chain": {
    "max_parallel": 10,
    "timeout_secs": 300,
    "enable_infinite": true
  },
  "scheduler": {
    "idle_interval_ms": 3600000,
    "active_interval_ms": 30000,
    "burst_interval_ms": 5000
  }
}
```

### REST API

When running, OMNI-HIVE exposes a REST API on port 7777:

```bash
# Health check
curl http://localhost:7777/health

# System status
curl http://localhost:7777/status

# List agents
curl http://localhost:7777/agents

# Spawn agent
curl -X POST http://localhost:7777/agents/spawn \
  -H "Content-Type: application/json" \
  -d '{"name": "analyzer", "type": "reasoning"}'

# Trigger evolution
curl -X POST http://localhost:7777/agents/evolve

# Brain usage
curl http://localhost:7777/brain/usage

# Portal search
curl -X POST http://localhost:7777/portal/search \
  -H "Content-Type: application/json" \
  -d '{"query": "code review discussion"}'
```

## Architecture

```
                              OMNI-HIVE DAEMON
    +------------------------------------------------------------------+
    |                      PANTHEON (GodAI Control)                     |
    |   [All-Seeing] [All-Knowing] [All-Controlling] [Predictive]      |
    +------------------------------------------------------------------+
                                    |
    +------------------------------------------------------------------+
    |                  TOKEN-AWARE BRAIN (Never Compact)                |
    |        5% System | 60% Context | 25% Response | 10% Reserve       |
    +------------------------------------------------------------------+
                                    |
    +------------------------------------------------------------------+
    |                    HYPERBEAST 3D MATRIX                           |
    |     N-Dimensional Hypercube with Wormhole Network                 |
    |     X: Models | Y: Time (infinite) | Z: Reasoning Depth           |
    +------------------------------------------------------------------+
                                    |
    +------------------------------------------------------------------+
    |                 MACRO-AGENT ORCHESTRATION                         |
    |         Macro -> Micro -> Nano (Dynamic MITOSIS/APOPTOSIS)        |
    +------------------------------------------------------------------+
                                    |
    +------------------------------------------------------------------+
    |                     CONTINUOUS CHAIN                              |
    |        .then() -> .parallel() -> .loop_until() -> .forever()      |
    +------------------------------------------------------------------+
                                    |
    +------------------------------------------------------------------+
    |                      MEGA-ADAPTER                                 |
    |    [Web] [Desktop] [API] [Local] [Multimodal] - 100+ platforms    |
    +------------------------------------------------------------------+
```

### Crate Structure

| Crate | Purpose |
|-------|---------|
| `omni-core` | Main daemon, CLI, REST API, scheduler |
| `omni-brain` | Token-aware context management |
| `omni-chain` | Hyperbeast matrix, chain orchestration, macro-agents |
| `omni-portal` | AI re-entry system, semantic search |
| `omni-ir` | Self-writing IR engine, auto-healing |
| `omni-terminal` | GPU-accelerated terminal (wgpu) |
| `omni-studio` | AI Studio GUI factory |
| `omni-integrations` | Platform adapters |

## Deployment Modes

| Mode | Memory | CPU | Use Case |
|------|--------|-----|----------|
| **Full** | 200MB+ | 5-10% | Desktop with browser UI |
| **Headless** | 50-100MB | 2-5% | Server, VPS, containers |
| **Supermini** | 2-12MB | 0.1% | Raspberry Pi, embedded, minimal VPS |

## Development

### Running Tests

```bash
cargo test --all
```

### Building Documentation

```bash
cargo doc --no-deps --open
```

### Code Style

```bash
cargo fmt --all
cargo clippy --all-targets
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting PRs.

## Acknowledgments

- Built with Rust and the amazing ecosystem (tokio, axum, chromiumoxide)
- Inspired by the need for unified AI orchestration
