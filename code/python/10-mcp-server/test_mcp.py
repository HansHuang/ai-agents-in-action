"""MCP System Tests

Tests cover:
  - MCP weather server  (tool listing, schemas, execution, error handling, resources)
  - MCP agent           (tool discovery, execution, multi-server, disconnect handling)
  - ServerRegistry      (namespacing, health checks, reconnection)
  - SimpleMCPServer     (schema generation, decorator registration)

The server tests start actual subprocesses via StdioServerParameters.
LLM calls in agent tests are mocked to avoid network dependencies.

Run::

    pip install mcp pytest pytest-asyncio
    pytest test_mcp.py -v
"""
from __future__ import annotations

import json
import pathlib
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).parent
WEATHER_SERVER = HERE / "weather_mcp_server" / "server.py"
SIMPLE_SERVER  = HERE / "simple_mcp_server.py"

# ---------------------------------------------------------------------------
# Guard: skip entire module if mcp is not installed
# ---------------------------------------------------------------------------
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MCP_AVAILABLE,
    reason="mcp package not installed — run: pip install mcp",
)

# ---------------------------------------------------------------------------
# Make local modules importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def weather_server_params() -> "StdioServerParameters":
    return StdioServerParameters(
        command=sys.executable,
        args=[str(WEATHER_SERVER)],
    )


@pytest.fixture
def simple_server_params() -> "StdioServerParameters":
    return StdioServerParameters(
        command=sys.executable,
        args=[str(SIMPLE_SERVER)],
    )


@asynccontextmanager
async def mcp_session(
    params: "StdioServerParameters",
) -> AsyncGenerator["ClientSession", None]:
    """Context manager that opens an MCP session to a server subprocess."""
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        yield session


# ===========================================================================
# 1–5: MCP Server Tests
# ===========================================================================

class TestMCPServer:
    """Exercise the weather MCP server directly over the MCP protocol."""

    @pytest.mark.asyncio
    async def test_server_lists_tools(self, weather_server_params):
        """Server must advertise get_weather and get_forecast with required fields."""
        async with mcp_session(weather_server_params) as session:
            result = await session.list_tools()
            names = [t.name for t in result.tools]

            assert "get_weather" in names,  f"Expected 'get_weather' in {names}"
            assert "get_forecast" in names, f"Expected 'get_forecast' in {names}"

            for tool in result.tools:
                assert tool.name,        f"Tool missing name"
                assert tool.description, f"Tool {tool.name!r} missing description"
                assert tool.inputSchema, f"Tool {tool.name!r} missing inputSchema"

    @pytest.mark.asyncio
    async def test_server_tool_schema_is_valid_json_schema(self, weather_server_params):
        """The get_weather inputSchema must follow JSON Schema conventions."""
        async with mcp_session(weather_server_params) as session:
            result = await session.list_tools()
            tool = next(t for t in result.tools if t.name == "get_weather")
            schema = tool.inputSchema

            assert schema["type"] == "object"
            assert "properties" in schema
            assert "city" in schema["properties"]
            assert "units" in schema["properties"]
            assert "required" in schema
            assert "city" in schema["required"]

            city_schema = schema["properties"]["city"]
            assert "type" in city_schema
            assert "description" in city_schema

            units_schema = schema["properties"]["units"]
            assert units_schema.get("type") == "string"
            assert "enum" in units_schema
            assert set(units_schema["enum"]) == {"celsius", "fahrenheit"}

    @pytest.mark.asyncio
    async def test_server_tool_call_returns_valid_response(self, weather_server_params):
        """Calling get_weather returns temperature, humidity, and condition fields."""
        async with mcp_session(weather_server_params) as session:
            result = await session.call_tool(
                "get_weather", {"city": "Tokyo", "units": "celsius"}
            )
            assert result.content, "Response must have content"
            data = json.loads(result.content[0].text)

            assert "temperature" in data, f"Missing 'temperature' in {data}"
            assert "humidity"    in data, f"Missing 'humidity' in {data}"
            assert "condition"   in data, f"Missing 'condition' in {data}"
            assert data["units"] == "celsius"

    @pytest.mark.asyncio
    async def test_server_tool_call_with_invalid_args_returns_error(
        self, weather_server_params
    ):
        """An unknown city returns a JSON error dict — not a server crash."""
        async with mcp_session(weather_server_params) as session:
            result = await session.call_tool(
                "get_weather", {"city": "XxNotARealCityXx"}
            )
            assert result.content
            data = json.loads(result.content[0].text)
            assert "error" in data, f"Expected 'error' key, got {data}"

    @pytest.mark.asyncio
    async def test_server_tool_call_missing_required_arg_returns_error(
        self, weather_server_params
    ):
        """Calling get_weather without 'city' returns an error, not a crash."""
        async with mcp_session(weather_server_params) as session:
            result = await session.call_tool("get_weather", {})
            assert result.content
            data = json.loads(result.content[0].text)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_server_resources_accessible(self, weather_server_params):
        """Reading weather://status returns a healthy status payload."""
        async with mcp_session(weather_server_params) as session:
            resources_result = await session.list_resources()
            uris = [str(r.uri) for r in resources_result.resources]
            assert "weather://status" in uris, (
                f"Expected 'weather://status' in {uris}"
            )

            status_result = await session.read_resource("weather://status")
            # SDK returns ReadResourceResult; contents is a list
            raw = status_result.contents[0].text
            data = json.loads(raw)
            assert data.get("status") == "healthy"
            assert "tools" in data
            assert "supported_cities" in data


# ===========================================================================
# 6–10: MCP Agent Tests
# ===========================================================================

class TestMCPAgent:
    """Tests for MCPAgent: tool discovery, execution, disconnect handling."""

    @pytest.mark.asyncio
    async def test_agent_discovers_tools_on_connect(self, weather_server_params):
        """After connect_server(), agent.tools includes weather__get_weather."""
        from mcp_agent import LLMProvider, MCPAgent

        agent = MCPAgent(llm_provider=MagicMock(spec=LLMProvider))
        try:
            count = await agent.connect_server("weather", weather_server_params)
            assert count > 0

            tool_names = [t["function"]["name"] for t in agent.tools]
            assert "weather__get_weather"  in tool_names
            assert "weather__get_forecast" in tool_names
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_agent_calls_mcp_tool(self, weather_server_params):
        """Agent executes a tool via MCP when the mocked LLM requests it."""
        from mcp_agent import AgentResult, MCPAgent

        call_count = 0

        async def mock_chat(messages, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "weather__get_weather",
                            "arguments": json.dumps({"city": "Tokyo"}),
                        },
                    }],
                }
            return {
                "role": "assistant",
                "content": "The weather in Tokyo is partly cloudy.",
                "tool_calls": [],
            }

        mock_llm = MagicMock()
        mock_llm.chat = mock_chat

        agent = MCPAgent(llm_provider=mock_llm)
        try:
            await agent.connect_server("weather", weather_server_params)
            result = await agent.run("What's the weather in Tokyo?")

            assert isinstance(result, AgentResult)
            assert "weather__get_weather" in result.tools_called
            assert result.answer
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_agent_uses_multiple_servers(
        self, weather_server_params, simple_server_params
    ):
        """Tools from both servers are collected and namespaced correctly."""
        from mcp_agent import LLMProvider, MCPAgent

        agent = MCPAgent(llm_provider=MagicMock(spec=LLMProvider))
        try:
            await agent.connect_server("weather", weather_server_params)
            await agent.connect_server("demo",    simple_server_params)

            tool_names = [t["function"]["name"] for t in agent.tools]

            # Every tool must be namespaced
            for name in tool_names:
                assert "__" in name, f"Tool '{name}' is not namespaced"

            prefixes = {name.split("__")[0] for name in tool_names}
            assert "weather" in prefixes
            assert "demo"    in prefixes
        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_agent_handles_server_disconnect(self, weather_server_params):
        """Agent returns a graceful result even when the server is gone."""
        from mcp_agent import AgentResult, MCPAgent

        call_count = 0

        async def mock_chat(messages, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_002",
                        "type": "function",
                        "function": {
                            "name": "weather__get_weather",
                            "arguments": json.dumps({"city": "Tokyo"}),
                        },
                    }],
                }
            return {
                "role": "assistant",
                "content": "Sorry, I couldn't retrieve the weather.",
                "tool_calls": [],
            }

        mock_llm = MagicMock()
        mock_llm.chat = mock_chat

        agent = MCPAgent(llm_provider=mock_llm)
        await agent.connect_server("weather", weather_server_params)

        # Simulate disconnect
        await agent.disconnect_server("weather")

        # Agent must not raise — it should handle the error and return
        result = await agent.run("What's the weather in Tokyo?")
        assert isinstance(result, AgentResult)
        # The tool execution error is propagated back to the LLM as a tool message

    @pytest.mark.asyncio
    async def test_agent_reconnect_after_disconnect(self, weather_server_params):
        """After reconnecting, agent can call tools again."""
        from mcp_agent import LLMProvider, MCPAgent

        agent = MCPAgent(llm_provider=MagicMock(spec=LLMProvider))
        await agent.connect_server("weather", weather_server_params)

        # Disconnect
        await agent.disconnect_server("weather")
        assert "weather" not in agent.server_registry.servers

        # Reconnect
        success = await agent.server_registry.reconnect("weather")
        assert success

        # Rebuild tool list after reconnect
        agent.tools = agent.server_registry.get_all_tools("openai")

        tool_names = [t["function"]["name"] for t in agent.tools]
        assert "weather__get_weather" in tool_names

        await agent.close()


# ===========================================================================
# 11–12: ServerRegistry Tests
# ===========================================================================

class TestServerRegistry:
    """Tests for ServerRegistry: namespacing and health checks."""

    @pytest.mark.asyncio
    async def test_registry_namespaces_tools(
        self, weather_server_params, simple_server_params
    ):
        """Every tool name is prefixed with its server name."""
        from mcp_agent import ServerRegistry

        registry = ServerRegistry()
        try:
            await registry.connect("weather", weather_server_params)
            await registry.connect("demo",    simple_server_params)

            all_tools = registry.get_all_tools("openai")
            names = [t["function"]["name"] for t in all_tools]

            for name in names:
                assert "__" in name, f"Tool '{name}' lacks server namespace"
                prefix = name.split("__")[0]
                assert prefix in ("weather", "demo"), (
                    f"Unexpected prefix: {prefix!r}"
                )
        finally:
            await registry.disconnect_all()

    @pytest.mark.asyncio
    async def test_registry_health_check(
        self, weather_server_params, simple_server_params
    ):
        """Health check reports correct connected/disconnected status."""
        from mcp_agent import ServerRegistry

        registry = ServerRegistry()
        await registry.connect("weather", weather_server_params)
        await registry.connect("demo",    simple_server_params)

        # Manually mark demo as disconnected
        registry.servers["demo"].connected = False

        health = registry.health_check()
        assert health["weather"]["connected"] is True
        assert health["demo"]["connected"]    is False

        await registry.disconnect_all()


# ===========================================================================
# 13–14: SimpleMCPServer Tests
# ===========================================================================

class TestSimpleMCPServer:
    """Tests for SimpleMCPServer: schema generation and decorator registration."""

    def test_auto_schema_generation(self):
        """Schema generated from type hints has correct types and required fields."""
        from simple_mcp_server import _generate_schema

        def greet(name: str, times: int) -> str:
            """Greet someone multiple times."""
            return f"Hello {name}! " * times

        schema = _generate_schema(greet)

        assert schema["type"] == "object"
        assert "name"  in schema["properties"]
        assert "times" in schema["properties"]
        assert schema["properties"]["name"]["type"]  == "string"
        assert schema["properties"]["times"]["type"] == "integer"
        assert set(schema["required"]) == {"name", "times"}

    def test_auto_schema_optional_not_required(self):
        """Optional parameters are absent from the required list."""
        from simple_mcp_server import _generate_schema
        from typing import Optional

        def search(query: str, limit: Optional[int] = None) -> str:
            """Search for items."""
            return query

        schema = _generate_schema(search)
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]

    def test_decorator_registers_tool(self):
        """@server.tool() registers the function and its auto-generated schema."""
        from simple_mcp_server import SimpleMCPServer

        srv = SimpleMCPServer("test-server")

        @srv.tool()
        def my_tool(value: int) -> int:
            """Double a value."""
            return value * 2

        assert "my_tool" in srv._tools
        tool_obj, func = srv._tools["my_tool"]
        assert tool_obj.name == "my_tool"
        assert "Double a value" in tool_obj.description
        assert tool_obj.inputSchema["properties"]["value"]["type"] == "integer"

    def test_decorator_custom_name_and_description(self):
        """Name and description overrides take precedence over the function."""
        from simple_mcp_server import SimpleMCPServer

        srv = SimpleMCPServer("test-server")

        @srv.tool(name="renamed_tool", description="Custom description.")
        def _func(x: str) -> str:
            """Original docstring."""
            return x

        assert "renamed_tool" in srv._tools
        assert "_func" not in srv._tools
        tool_obj, _ = srv._tools["renamed_tool"]
        assert tool_obj.description == "Custom description."

    def test_resource_decorator_registers_resource(self):
        """@server.resource() registers a resource accessible by URI."""
        from simple_mcp_server import SimpleMCPServer

        srv = SimpleMCPServer("test-server")

        @srv.resource("config://version")
        def version_info() -> str:
            return '{"version": "1.0.0"}'

        assert "config://version" in srv._resources

    @pytest.mark.asyncio
    async def test_simple_server_lists_tools_via_mcp(self, simple_server_params):
        """Tools registered with @server.tool() are discoverable over MCP."""
        async with mcp_session(simple_server_params) as session:
            result = await session.list_tools()
            names = [t.name for t in result.tools]

            # Demo server registers: hello, calculate, search_files, send_notification
            assert "hello"             in names
            assert "calculate"         in names
            assert "search_files"      in names
            assert "send_notification" in names
