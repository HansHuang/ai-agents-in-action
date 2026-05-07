# 10 — Model Context Protocol (MCP)

> **"MCP is the USB-C of AI agents. One protocol, any tool, any agent."**

This folder demonstrates the MCP standard for agent-tool communication.
See → [docs/05-the-tool-ecosystem/04-mcp-protocol.md](../../../docs/05-the-tool-ecosystem/04-mcp-protocol.md)

Cross-references: [Node.js](../../nodejs/10-mcp-server/) · [Go](../../go/10-mcp-server/)

---

## Architecture

```
MCP Client (Agent)
       │
       │  MCP Protocol (JSON-RPC over stdio or HTTP)
       │
  ┌────┴──────────────────┐
  │  MCP Server (weather)  │  tools: get_weather, get_forecast
  │  MCP Server (demo)     │  tools: hello, calculate, …
  └───────────────────────┘
```

---

## What's in This Folder

| File / Folder | Purpose |
|---|---|
| `weather_mcp_server/` | Complete MCP weather server — tools, resources, rate-limit stub |
| `simple_mcp_server.py` | Decorator-based API: build MCP servers in ~10 lines |
| `mcp_agent.py` | Agent that discovers and uses tools from MCP servers |
| `mcp_marketplace.py` | Discover, catalog, and test MCP servers |
| `test_mcp.py` | 14 pytest integration tests |

---

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the weather server (another terminal)
cd weather_mcp_server
python server.py

# 3. Connect the agent (back in first terminal)
python mcp_agent.py            # requires OPENAI_API_KEY

# 4. Build and run your own server
python simple_mcp_server.py    # demo: hello, calculate, search_files
```

---

## Pre-Built MCP Servers

Connect to these without writing any server code:

```bash
# Filesystem  — read, write, search files
npx -y @modelcontextprotocol/server-filesystem /path/to/dir

# GitHub      — issues, PRs, branches
npx -y @modelcontextprotocol/server-github

# Postgres    — query with schema awareness
npx -y @modelcontextprotocol/server-postgres $DATABASE_URL
```

Use `mcp_marketplace.py` to discover more servers.

---

## Run Tests

```bash
pytest test_mcp.py -v
```
