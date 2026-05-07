"""MCP Tool Marketplace — discover, catalog, test, and configure MCP servers.

Discovers servers from three sources:
  - Local directories  (Python / TypeScript / JavaScript files)
  - npm registry       (@modelcontextprotocol/* packages)
  - PyPI               (mcp-server-* packages)

Usage::

    python mcp_marketplace.py
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ServerInfo:
    name: str
    description: str
    source: str               # "npm" | "pypi" | "local"
    install_command: str
    run_command: list[str]    # argv to start the server
    tool_count: int = 0
    categories: list[str] = field(default_factory=list)
    rating: float | None = None
    version: str = "unknown"


# ---------------------------------------------------------------------------
# Curated npm catalog  (Anthropic + community servers)
# ---------------------------------------------------------------------------

_NPM_CATALOG: list[ServerInfo] = [
    ServerInfo(
        name="filesystem",
        description="Read, write, search, and move files and directories.",
        source="npm",
        install_command="npm install -g @modelcontextprotocol/server-filesystem",
        run_command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        tool_count=9,
        categories=["files", "utilities"],
        rating=4.8,
    ),
    ServerInfo(
        name="github",
        description="GitHub repository management: issues, PRs, branches, files.",
        source="npm",
        install_command="npm install -g @modelcontextprotocol/server-github",
        run_command=["npx", "-y", "@modelcontextprotocol/server-github"],
        tool_count=24,
        categories=["development", "git"],
        rating=4.7,
    ),
    ServerInfo(
        name="postgres",
        description="PostgreSQL database querying with schema awareness.",
        source="npm",
        install_command="npm install -g @modelcontextprotocol/server-postgres",
        run_command=["npx", "-y", "@modelcontextprotocol/server-postgres", "$DATABASE_URL"],
        tool_count=3,
        categories=["database"],
        rating=4.6,
    ),
    ServerInfo(
        name="slack",
        description="Send Slack messages and read channel history.",
        source="npm",
        install_command="npm install -g @modelcontextprotocol/server-slack",
        run_command=["npx", "-y", "@modelcontextprotocol/server-slack"],
        tool_count=5,
        categories=["communication"],
        rating=4.3,
    ),
    ServerInfo(
        name="brave-search",
        description="Web and local search via the Brave Search API.",
        source="npm",
        install_command="npm install -g @modelcontextprotocol/server-brave-search",
        run_command=["npx", "-y", "@modelcontextprotocol/server-brave-search"],
        tool_count=2,
        categories=["search", "web"],
        rating=4.5,
    ),
    ServerInfo(
        name="puppeteer",
        description="Browser automation: navigate, screenshot, click, fill forms.",
        source="npm",
        install_command="npm install -g @modelcontextprotocol/server-puppeteer",
        run_command=["npx", "-y", "@modelcontextprotocol/server-puppeteer"],
        tool_count=7,
        categories=["browser", "automation"],
        rating=4.4,
    ),
    ServerInfo(
        name="memory",
        description="Persistent knowledge graph for long-term agent memory.",
        source="npm",
        install_command="npm install -g @modelcontextprotocol/server-memory",
        run_command=["npx", "-y", "@modelcontextprotocol/server-memory"],
        tool_count=6,
        categories=["memory", "knowledge"],
        rating=4.6,
    ),
    ServerInfo(
        name="git",
        description="Git repository operations: log, diff, commit, branch.",
        source="npm",
        install_command="npm install -g @modelcontextprotocol/server-git",
        run_command=["npx", "-y", "@modelcontextprotocol/server-git"],
        tool_count=11,
        categories=["development", "git"],
        rating=4.5,
    ),
    ServerInfo(
        name="sqlite",
        description="SQLite database operations with business intelligence tools.",
        source="npm",
        install_command="npm install -g @modelcontextprotocol/server-sqlite",
        run_command=["npx", "-y", "@modelcontextprotocol/server-sqlite"],
        tool_count=8,
        categories=["database"],
        rating=4.4,
    ),
]

# Curated PyPI catalog
_PYPI_CATALOG: list[ServerInfo] = [
    ServerInfo(
        name="mcp-server-fetch",
        description="Fetch and extract content from web URLs.",
        source="pypi",
        install_command="pip install mcp-server-fetch",
        run_command=["python", "-m", "mcp_server_fetch"],
        tool_count=1,
        categories=["web", "utilities"],
        rating=4.2,
    ),
    ServerInfo(
        name="mcp-server-time",
        description="Get current time and perform timezone conversions.",
        source="pypi",
        install_command="pip install mcp-server-time",
        run_command=["python", "-m", "mcp_server_time"],
        tool_count=2,
        categories=["utilities"],
        rating=4.1,
    ),
]

# Local discovery: file patterns → description
_LOCAL_PATTERNS: list[tuple[str, str]] = [
    ("*server*.py",  "Python MCP server"),
    ("*mcp*.py",     "Python MCP module"),
    ("server.ts",    "TypeScript MCP server"),
    ("server.js",    "JavaScript MCP server"),
]


# ---------------------------------------------------------------------------
# JSON Schema validator  (no external deps)
# ---------------------------------------------------------------------------

def _validate_json_schema(schema: dict, tool_name: str) -> list[str]:
    """Return a list of validation error strings for *schema*."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return [f"{tool_name}: inputSchema must be a dict"]
    if schema.get("type") != "object":
        errors.append(
            f"{tool_name}: schema.type must be 'object', got {schema.get('type')!r}"
        )
    props = schema.get("properties")
    if props is None:
        errors.append(f"{tool_name}: schema missing 'properties'")
    elif not isinstance(props, dict):
        errors.append(f"{tool_name}: schema.properties must be a dict")
    else:
        for prop_name, prop_schema in props.items():
            if not isinstance(prop_schema, dict):
                errors.append(f"{tool_name}.{prop_name}: property schema must be a dict")
            elif "type" not in prop_schema:
                errors.append(f"{tool_name}.{prop_name}: missing 'type'")
    return errors


# ---------------------------------------------------------------------------
# MCPMarketplace
# ---------------------------------------------------------------------------

class MCPMarketplace:
    """Discover and manage MCP servers from multiple sources."""

    def __init__(self) -> None:
        self.available_servers: dict[str, ServerInfo] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_local(self, search_paths: list[str]) -> list[ServerInfo]:
        """Find MCP server files in local directories.

        Searches recursively for Python, TypeScript, and JavaScript files
        whose names suggest they are MCP servers.
        """
        import glob

        found: list[ServerInfo] = []
        for search_path in search_paths:
            for pattern, description in _LOCAL_PATTERNS:
                glob_pattern = os.path.join(search_path, "**", pattern)
                for fpath in glob.glob(glob_pattern, recursive=True):
                    server_name = Path(fpath).stem
                    if server_name in self.available_servers:
                        continue
                    run_cmd = (
                        [sys.executable, fpath]
                        if fpath.endswith(".py")
                        else ["node", fpath]
                    )
                    info = ServerInfo(
                        name=server_name,
                        description=f"{description}: {fpath}",
                        source="local",
                        install_command="(already installed)",
                        run_command=run_cmd,
                        categories=["local"],
                    )
                    self.available_servers[server_name] = info
                    found.append(info)
        return found

    def discover_npm(self) -> list[ServerInfo]:
        """Return the curated list of npm MCP servers."""
        for s in _NPM_CATALOG:
            self.available_servers[s.name] = s
        return list(_NPM_CATALOG)

    def discover_pypi(self) -> list[ServerInfo]:
        """Return the curated list of PyPI MCP servers."""
        for s in _PYPI_CATALOG:
            self.available_servers[s.name] = s
        return list(_PYPI_CATALOG)

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    def install_server(self, server_name: str, source: str = "npm") -> bool:
        """Install an MCP server from a package registry.

        Returns True on success.
        """
        info = self.available_servers.get(server_name)
        if info is None:
            print(f"[marketplace] Unknown server: {server_name!r}", file=sys.stderr)
            return False

        print(f"[marketplace] Running: {info.install_command}", file=sys.stderr)
        try:
            subprocess.run(
                info.install_command.split(),
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            print(f"[marketplace] Installed '{server_name}' ✓", file=sys.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[marketplace] Install failed: {exc.stderr}", file=sys.stderr)
            return False
        except subprocess.TimeoutExpired:
            print(f"[marketplace] Install timed out for '{server_name}'.", file=sys.stderr)
            return False

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    async def test_server(self, server_name: str) -> dict:
        """Connect to a server and verify its tools and schemas.

        Checks:
        - Connection succeeds
        - ``list_tools()`` returns valid tool objects
        - Each tool's ``inputSchema`` is valid JSON Schema
        """
        info = self.available_servers.get(server_name)
        if info is None:
            return {"server": server_name, "status": "error", "error": "Unknown server"}

        result: dict[str, Any] = {
            "server": server_name,
            "source": info.source,
            "status": "unknown",
            "connected": False,
            "tools": [],
            "tool_count": 0,
            "schema_errors": [],
        }

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=info.run_command[0],
                args=info.run_command[1:],
            )
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                result["connected"] = True
                tools_result = await session.list_tools()
                tools = tools_result.tools

                result["tools"] = [t.name for t in tools]
                result["tool_count"] = len(tools)

                for tool in tools:
                    errs = _validate_json_schema(tool.inputSchema, tool.name)
                    result["schema_errors"].extend(errs)

                result["status"] = "healthy" if not result["schema_errors"] else "schema_errors"

        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)

        return result

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def generate_config(self, server_names: list[str]) -> dict:
        """Generate an agent configuration dict for the selected servers.

        The output is compatible with ``MCPAgent`` (from mcp_agent.py).
        """
        config: dict = {"servers": []}
        for name in server_names:
            info = self.available_servers.get(name)
            if info is None:
                print(
                    f"[marketplace] Warning: unknown server {name!r}, skipping.",
                    file=sys.stderr,
                )
                continue
            config["servers"].append({
                "name": name,
                "command": info.run_command[0],
                "args": info.run_command[1:],
                "source": info.source,
                "auto_connect": True,
            })
        return config

    # ------------------------------------------------------------------
    # Listing / catalog
    # ------------------------------------------------------------------

    def list_servers(self, filter: str | None = None) -> list[ServerInfo]:
        """List available servers, optionally filtered by category or source."""
        servers = list(self.available_servers.values())
        if filter:
            servers = [
                s for s in servers
                if filter in s.categories or filter == s.source
            ]
        return sorted(servers, key=lambda s: (s.source, s.name))

    def print_catalog(self, filter: str | None = None) -> None:
        """Print a formatted catalog to stderr."""
        servers = self.list_servers(filter)
        print(f"\n{'=' * 72}", file=sys.stderr)
        print(f"  MCP Server Catalog  ({len(servers)} servers)", file=sys.stderr)
        print(f"{'=' * 72}", file=sys.stderr)

        current_source: str | None = None
        for info in servers:
            if info.source != current_source:
                current_source = info.source
                print(f"\n  [{info.source.upper()}]", file=sys.stderr)

            rating_str = f"★ {info.rating:.1f}" if info.rating else "     "
            cats = ", ".join(info.categories)
            count_str = f"{info.tool_count} tools" if info.tool_count else "? tools"
            print(
                f"  {info.name:<32} {rating_str}  {count_str:<12}  [{cats}]",
                file=sys.stderr,
            )
            print(f"    {info.description}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def run_demo() -> None:
    marketplace = MCPMarketplace()

    print("\n[1] Discovering servers ...", file=sys.stderr)

    here = str(Path(__file__).parent)
    local = marketplace.discover_local([here])
    print(f"  Local  : {len(local)} server(s) found", file=sys.stderr)

    npm_servers = marketplace.discover_npm()
    print(f"  npm    : {len(npm_servers)} server(s) cataloged", file=sys.stderr)

    pypi_servers = marketplace.discover_pypi()
    print(f"  PyPI   : {len(pypi_servers)} server(s) cataloged", file=sys.stderr)

    marketplace.print_catalog()

    # Test the local weather server if present
    weather_path = Path(__file__).parent / "weather_mcp_server" / "server.py"
    if weather_path.exists():
        print("\n[2] Testing local weather server ...", file=sys.stderr)
        test_result = await marketplace.test_server("server")
        print(json.dumps(test_result, indent=2), file=sys.stderr)
    else:
        print(
            "\n[2] Weather server not found — run from code/python/10-mcp-server/.",
            file=sys.stderr,
        )

    print("\n[3] Generating config for 3 npm servers ...", file=sys.stderr)
    config = marketplace.generate_config(["filesystem", "github", "memory"])
    print(json.dumps(config, indent=2), file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(run_demo())
