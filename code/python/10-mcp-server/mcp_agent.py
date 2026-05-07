"""MCP-Enabled Agent — discovers and calls tools from MCP servers.

Architecture:
  MCPAgent        high-level agent: connect, run, close
  ServerRegistry  manages multiple MCP server connections
  ServerConnection  one long-lived connection to one MCP server

The tool namespace pattern is {server_name}__{tool_name}, so two servers
that both expose a "search" tool never collide.

Usage:
    python mcp_agent.py            # runs the built-in demo
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Returned by MCPAgent.run()."""
    answer: str
    messages: list[dict]
    tools_called: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Result from one MCP tool execution."""
    server_name: str
    tool_name: str
    output: str
    is_error: bool = False


# ---------------------------------------------------------------------------
# LLM provider abstraction  (swap in Anthropic, Ollama, etc.)
# ---------------------------------------------------------------------------

class LLMProvider:
    """Base interface — override ``chat`` in each provider subclass."""

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI chat-completions with function calling."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        from openai import AsyncOpenAI  # lazy import
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (msg.tool_calls or [])
            ],
        }


# ---------------------------------------------------------------------------
# ServerConnection  — one long-lived MCP session
# ---------------------------------------------------------------------------

class ServerConnection:
    """Maintains a subprocess + MCP session for a single server."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.session: ClientSession | None = None
        self.tools: list = []          # list[mcp.types.Tool]
        self._exit_stack = AsyncExitStack()
        self.connected = False

    async def connect(self, params: StdioServerParameters) -> None:
        """Start the server subprocess and initialise the MCP session."""
        try:
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self.session.initialize()
            result = await self.session.list_tools()
            self.tools = result.tools
            self.connected = True
            print(
                f"[registry] Connected to '{self.name}': "
                f"{len(self.tools)} tools discovered",
                file=sys.stderr,
            )
        except Exception as exc:
            self.connected = False
            await self._exit_stack.aclose()
            raise RuntimeError(
                f"Failed to connect to MCP server '{self.name}': {exc}"
            ) from exc

    async def call_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute *tool_name* on this server and return a ToolResult."""
        if not self.connected or self.session is None:
            return ToolResult(
                server_name=self.name,
                tool_name=tool_name,
                output=json.dumps({"error": f"Server '{self.name}' is not connected."}),
                is_error=True,
            )
        try:
            result = await self.session.call_tool(tool_name, arguments)
            output = "\n".join(
                item.text for item in result.content if hasattr(item, "text")
            )
            return ToolResult(
                server_name=self.name,
                tool_name=tool_name,
                output=output or "(empty response)",
                is_error=bool(result.isError),
            )
        except Exception as exc:
            self.connected = False
            return ToolResult(
                server_name=self.name,
                tool_name=tool_name,
                output=json.dumps({"error": f"Tool execution failed: {exc}"}),
                is_error=True,
            )

    async def close(self) -> None:
        self.connected = False
        await self._exit_stack.aclose()


# ---------------------------------------------------------------------------
# ServerRegistry  — manages multiple ServerConnections
# ---------------------------------------------------------------------------

class ServerRegistry:
    """Connect to multiple MCP servers and route tool calls."""

    def __init__(self) -> None:
        self.servers: dict[str, ServerConnection] = {}
        # Preserved for reconnection
        self._params: dict[str, StdioServerParameters] = {}

    async def connect(
        self, name: str, params: StdioServerParameters
    ) -> list[dict]:
        """Connect to a server; return its tools in OpenAI format."""
        conn = ServerConnection(name)
        await conn.connect(params)
        self.servers[name] = conn
        self._params[name] = params
        return self._format_tools(conn, "openai")

    # ------------------------------------------------------------------
    # Tool format conversion
    # ------------------------------------------------------------------

    def _format_tools(self, conn: ServerConnection, fmt: str) -> list[dict]:
        return [self._to_openai(t, conn.name) if fmt == "openai"
                else self._to_anthropic(t, conn.name)
                for t in conn.tools]

    def _to_openai(self, mcp_tool, server_name: str) -> dict:
        return {
            "type": "function",
            "function": {
                "name": f"{server_name}__{mcp_tool.name}",
                "description": mcp_tool.description or "",
                "parameters": mcp_tool.inputSchema,
            },
        }

    def _to_anthropic(self, mcp_tool, server_name: str) -> dict:
        return {
            "name": f"{server_name}__{mcp_tool.name}",
            "description": mcp_tool.description or "",
            "input_schema": mcp_tool.inputSchema,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_tools(self, fmt: str = "openai") -> list[dict]:
        """Return tools from all *connected* servers, namespaced by server."""
        tools: list[dict] = []
        for conn in self.servers.values():
            if conn.connected:
                tools.extend(self._format_tools(conn, fmt))
        return tools

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict
    ) -> str:
        """Execute *tool_name* on *server_name* and return result text."""
        conn = self.servers.get(server_name)
        if conn is None:
            return json.dumps({"error": f"Unknown server: {server_name!r}"})
        result = await conn.call_tool(tool_name, arguments)
        return result.output

    async def reconnect(self, server_name: str) -> bool:
        """Reconnect to a server that disconnected. Returns True on success."""
        params = self._params.get(server_name)
        if params is None:
            return False

        old = self.servers.get(server_name)
        if old:
            await old.close()

        try:
            conn = ServerConnection(server_name)
            await conn.connect(params)
            self.servers[server_name] = conn
            print(f"[registry] Reconnected to '{server_name}'", file=sys.stderr)
            return True
        except Exception as exc:
            print(
                f"[registry] Reconnect failed for '{server_name}': {exc}",
                file=sys.stderr,
            )
            return False

    async def disconnect_all(self) -> None:
        """Close every server connection."""
        for conn in self.servers.values():
            await conn.close()
        self.servers.clear()

    def health_check(self) -> dict:
        """Return a connection-status snapshot for all registered servers."""
        return {
            name: {
                "connected": conn.connected,
                "tool_count": len(conn.tools),
                "tools": [t.name for t in conn.tools],
            }
            for name, conn in self.servers.items()
        }


# ---------------------------------------------------------------------------
# MCPAgent  — the main agent
# ---------------------------------------------------------------------------

class MCPAgent:
    """Provider-agnostic agent that uses MCP for tool discovery and execution.

    1. Call ``connect_server()`` for each MCP server to use.
    2. Call ``run(user_input)`` to start the agent loop.
    3. Call ``close()`` when done.
    """

    MAX_ITERATIONS = 10  # Guard against runaway loops

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm = llm_provider
        self.server_registry = ServerRegistry()
        self.tools: list[dict] = []  # OpenAI-format tool schemas

    async def connect_server(
        self, name: str, server_params: StdioServerParameters
    ) -> int:
        """Connect to an MCP server and register its tools.

        Returns the number of tools discovered.
        """
        new_tools = await self.server_registry.connect(name, server_params)
        self.tools.extend(new_tools)
        print(
            f"[agent] +{len(new_tools)} tools from '{name}'. "
            f"Total: {len(self.tools)}",
            file=sys.stderr,
        )
        return len(new_tools)

    async def disconnect_server(self, name: str) -> None:
        """Disconnect from *name* and remove its tools from the registry."""
        conn = self.server_registry.servers.get(name)
        if conn:
            await conn.close()
            del self.server_registry.servers[name]
        # Rebuild the tool list without the disconnected server
        self.tools = self.server_registry.get_all_tools("openai")
        print(f"[agent] Disconnected from '{name}'", file=sys.stderr)

    def _convert_mcp_tool_to_openai(self, mcp_tool, server_name: str) -> dict:
        """Convert an MCP tool schema to OpenAI function-calling format.

        Namespaces the tool as ``{server_name}__{tool_name}``.
        """
        return {
            "type": "function",
            "function": {
                "name": f"{server_name}__{mcp_tool.name}",
                "description": mcp_tool.description or "",
                "parameters": mcp_tool.inputSchema,
            },
        }

    def _convert_mcp_tool_to_anthropic(self, mcp_tool, server_name: str) -> dict:
        """Convert an MCP tool schema to Anthropic tool format."""
        return {
            "name": f"{server_name}__{mcp_tool.name}",
            "description": mcp_tool.description or "",
            "input_schema": mcp_tool.inputSchema,
        }

    async def run(self, user_input: str) -> AgentResult:
        """Run the agent loop with all MCP-discovered tools.

        The loop continues until the LLM returns a final answer
        (no tool_calls) or MAX_ITERATIONS is reached.
        """
        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Use the available tools whenever "
                    "needed to answer the user's question accurately."
                ),
            },
            {"role": "user", "content": user_input},
        ]
        tools_called: list[str] = []

        for _ in range(self.MAX_ITERATIONS):
            response = await self.llm.chat(
                messages=messages,
                tools=self.tools if self.tools else None,
            )
            messages.append(response)

            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                # No more tool calls — final answer
                return AgentResult(
                    answer=response.get("content") or "",
                    messages=messages,
                    tools_called=tools_called,
                )

            # Execute each tool call via MCP
            for tc in tool_calls:
                full_name = tc["function"]["name"]
                result = await self._execute_mcp_tool(
                    full_name,
                    json.loads(tc["function"]["arguments"]),
                )
                tools_called.append(full_name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result.output,
                })

        return AgentResult(
            answer="Maximum iterations reached without a final answer.",
            messages=messages,
            tools_called=tools_called,
        )

    async def _execute_mcp_tool(
        self, full_tool_name: str, arguments: dict
    ) -> ToolResult:
        """Parse ``{server_name}__{tool_name}`` and route to the correct server."""
        if "__" not in full_tool_name:
            return ToolResult(
                server_name="unknown",
                tool_name=full_tool_name,
                output=json.dumps({
                    "error": (
                        f"Invalid tool name format: {full_tool_name!r}. "
                        "Expected '{server_name}__{tool_name}'."
                    )
                }),
                is_error=True,
            )

        server_name, tool_name = full_tool_name.split("__", 1)
        print(
            f"[agent] → {server_name}/{tool_name} args={arguments}",
            file=sys.stderr,
        )
        output = await self.server_registry.call_tool(
            server_name, tool_name, arguments
        )
        return ToolResult(
            server_name=server_name,
            tool_name=tool_name,
            output=output,
        )

    async def close(self) -> None:
        """Disconnect from all MCP servers."""
        await self.server_registry.disconnect_all()
        print("[agent] All servers disconnected.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def run_demo() -> None:
    """Connect to the weather server and run several queries."""
    import pathlib

    here = pathlib.Path(__file__).parent
    weather_server = here / "weather_mcp_server" / "server.py"

    if not weather_server.exists():
        print(
            f"[demo] Weather server not found at {weather_server}.\n"
            "Run from code/python/10-mcp-server/.",
            file=sys.stderr,
        )
        return

    print("=" * 60, file=sys.stderr)
    print("MCP Agent Demo", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Use OpenAIProvider — requires OPENAI_API_KEY env var
    agent = MCPAgent(llm_provider=OpenAIProvider(model="gpt-4o"))

    # Connect to the local weather server
    await agent.connect_server(
        "weather",
        StdioServerParameters(
            command=sys.executable,
            args=[str(weather_server)],
        ),
    )

    # Print discovered tool schemas
    print("\n--- Discovered Tools ---", file=sys.stderr)
    for tool in agent.tools:
        print(f"  {tool['function']['name']}", file=sys.stderr)

    print("\n--- Tool Schemas ---", file=sys.stderr)
    for tool in agent.tools:
        print(
            f"\n{tool['function']['name']}:\n"
            + json.dumps(tool["function"]["parameters"], indent=2),
            file=sys.stderr,
        )

    # Run queries
    queries = [
        "What's the weather in Tokyo?",
        "What's the weather in London and in Dubai?",
        "Give me a 3-day forecast for Paris.",
    ]

    for query in queries:
        print(f"\n{'─' * 50}", file=sys.stderr)
        print(f"Query: {query}", file=sys.stderr)
        result = await agent.run(query)
        print(f"Answer: {result.answer}", file=sys.stderr)
        print(f"Tools called: {result.tools_called}", file=sys.stderr)

    # Health check
    print(
        f"\n--- Health Check ---\n"
        + json.dumps(agent.server_registry.health_check(), indent=2),
        file=sys.stderr,
    )

    # Demonstrate reconnection
    print("\n--- Reconnection Demo ---", file=sys.stderr)
    await agent.disconnect_server("weather")
    success = await agent.server_registry.reconnect("weather")
    print(f"Reconnected: {success}", file=sys.stderr)

    await agent.close()


if __name__ == "__main__":
    asyncio.run(run_demo())
