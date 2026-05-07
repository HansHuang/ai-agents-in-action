# The Model Context Protocol (MCP)

## What You'll Learn
- What MCP is and why it exists: the USB-C of AI agent integrations
- MCP architecture: clients, servers, and transports
- How MCP differs from direct function calling
- Building an MCP server that exposes tools, resources, and prompts
- Building an MCP client that discovers and calls remote tools
- The MCP ecosystem: pre-built servers and the emerging tool marketplace

## Prerequisites
- [Tool Design Patterns](../02-the-agent-loop/02-tool-design-patterns.md) — tools are what MCP exposes
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — agents are MCP clients
- [Model Providers](01-model-providers.md) — MCP is provider-agnostic

---

## The Integration Problem

Every LLM provider has a different function-calling API. Every tool you build is tightly coupled to your agent code. If you want to use the same weather tool with Claude, GPT-4o, and a local Llama model, you're implementing it three times.

```
┌─────────────────────────────────────────────────────┐
│                  WITHOUT MCP                         │
│                                                      │
│  Agent A (OpenAI) ─── get_weather() (Python)         │
│  Agent B (Claude) ─── get_weather() (TypeScript)     │
│  Agent C (Llama)  ─── get_weather() (Go)             │
│                                                      │
│  Same tool, three implementations, three languages.  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   WITH MCP                           │
│                                                      │
│  Agent A (OpenAI) ─┐                                │
│  Agent B (Claude) ─┼── MCP Server (weather)          │
│  Agent C (Llama)  ─┘   One implementation            │
│                        Any language                  │
│                        Any agent                     │
└─────────────────────────────────────────────────────┘
```

MCP solves this. It's an open protocol from Anthropic that standardizes how agents discover and use tools — regardless of which LLM provider, which programming language, or which framework you're using.

---

## What MCP Standardizes

MCP defines three primitives that a server can expose:

| Primitive | What It Is | Example |
|:---|:---|:---|
| **Tools** | Functions the agent can call | `get_weather`, `search_database`, `send_email` |
| **Resources** | Data the agent can read | File contents, database records, API responses |
| **Prompts** | Reusable prompt templates | "Summarize this document," "Translate to French" |

And two transports for communication:

| Transport | How It Works | Best For |
|:---|:---|:---|
| **stdio** | Standard input/output | Local tools, CLI integration |
| **Streamable HTTP** | HTTP with optional Server-Sent Events fallback | Remote tools, network services |

> Note: The original MCP spec used HTTP+SSE. MCP 1.x replaces it with Streamable HTTP, which carries the same semantics but works over standard HTTP POST and streams responses progressively. Most SDKs handle this transparently.

---

## MCP Architecture

```
┌──────────────────────┐
│    MCP CLIENT        │  ← Your agent or application
│  (Agent, IDE, App)   │
└──────────┬───────────┘
           │ MCP Protocol (JSON-RPC over stdio or HTTP)
           │
    ┌──────┼──────┬──────────┐
    ▼      ▼      ▼          ▼
┌──────┐┌──────┐┌──────┐┌──────┐
│MCP   ││MCP   ││MCP   ││MCP   │
│Server││Server││Server││Server│
│      ││      ││      ││      │
│Weather│Files │Database│Web  │
└──────┘└──────┘└──────┘└──────┘
```

### The Lifecycle

1. **Client connects** to a server via stdio or HTTP
2. **Client discovers** available tools, resources, and prompts
3. **Client calls** tools as needed, reads resources, uses prompts
4. **Server responds** with results
5. **Client disconnects** when done

---

## Building an MCP Server

An MCP server is a process that exposes tools via the MCP protocol. Here's a complete weather server:

```python
# weather_server.py
import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Create the MCP server
server = Server("weather-server")

# Define tools — the source of truth for schemas
@server.list_tools()
async def list_tools() -> list[Tool]:
    """Tell clients what tools are available."""
    return [
        Tool(
            name="get_weather",
            description="Get current weather conditions for a city. "
                        "Returns temperature, humidity, and conditions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name with country code. Example: 'Tokyo, JP'"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Defaults to celsius."
                    }
                },
                "required": ["city"]
            }
        ),
        Tool(
            name="get_forecast",
            description="Get 5-day weather forecast for a city.",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name with country code."}
                },
                "required": ["city"]
            }
        ),
    ]

# Implement the tools — return error content, never raise
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from clients."""
    if name == "get_weather":
        city = arguments.get("city")
        if not city:
            return [TextContent(type="text", text=json.dumps({"error": "Missing: city"}))]
        units = arguments.get("units", "celsius")
        # In production: call a real weather API
        weather_data = get_weather_data(city, units)
        return [TextContent(type="text", text=json.dumps(weather_data, indent=2))]

    elif name == "get_forecast":
        city = arguments.get("city")
        if not city:
            return [TextContent(type="text", text=json.dumps({"error": "Missing: city"}))]
        return [TextContent(type="text", text=json.dumps(get_forecast_data(city), indent=2))]

    # Return an error instead of raising — raising crashes the server
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]  

# Mock weather data (replace with real API)
def get_weather_data(city: str, units: str) -> dict:
    return {
        "city": city,
        "temperature": 22 if units == "celsius" else 72,
        "units": units,
        "humidity": 65,
        "condition": "partly cloudy",
        "timestamp": "2026-05-05T12:00:00Z"
    }

def get_forecast_data(city: str) -> dict:
    return {
        "city": city,
        "forecast": [
            {"day": "Mon", "high": 22, "low": 15, "condition": "sunny"},
            {"day": "Tue", "high": 20, "low": 14, "condition": "rain"},
            {"day": "Wed", "high": 23, "low": 16, "condition": "partly cloudy"},
            {"day": "Thu", "high": 25, "low": 17, "condition": "sunny"},
            {"day": "Fri", "high": 21, "low": 13, "condition": "cloudy"}
        ]
    }

# Run the server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),  # NOT InitializationCapabilities
        )

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
pip install mcp
python weather_server.py
```

The server is now waiting for connections on stdio. Any MCP client can connect and use its tools.

---

## Building an MCP Client

The client is your agent. It connects to MCP servers, discovers tools, and calls them:

```python
# mcp_agent.py
import asyncio
import json
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

class MCPAgent:
    """
    An agent that discovers and uses tools via MCP.
    Provider-agnostic: works with any LLM, any MCP server.
    """

    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.mcp_tools = []    # tools discovered from MCP servers
        self.sessions = {}     # name → ClientSession
        self._exit_stacks = {} # name → AsyncExitStack (keeps subprocess alive)

    async def connect_server(
        self, name: str, server_params: StdioServerParameters
    ) -> None:
        """
        Connect to an MCP server and discover its tools.
        Uses AsyncExitStack to keep the subprocess alive for the session's lifetime.
        """
        stack = AsyncExitStack()
        # stdio_client is a context manager — NOT an awaitable
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # list_tools() returns ListToolsResult; access .tools for the list
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            self.mcp_tools.append(self._to_openai(tool, name))

        self.sessions[name] = session
        self._exit_stacks[name] = stack
        print(f"Connected to '{name}': {len(tools_result.tools)} tools")

    def _to_openai(self, mcp_tool, server_name: str) -> dict:
        """Convert MCP tool schema to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": f"{server_name}__{mcp_tool.name}",  # namespace by server
                "description": mcp_tool.description,
                "parameters": mcp_tool.inputSchema,
            },
        }

    async def run(self, user_input: str) -> str:
        """Run the agent with MCP-discovered tools."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant with tools."},
            {"role": "user", "content": user_input},
        ]
        while True:
            response = await self.llm.chat(messages=messages, tools=self.mcp_tools or None)
            messages.append(response)
            if not response.get("tool_calls"):
                return response["content"]
            for tc in response["tool_calls"]:
                server_name, tool_name = tc["function"]["name"].split("__", 1)
                result = await self.sessions[server_name].call_tool(
                    tool_name,
                    arguments=json.loads(tc["function"]["arguments"]),
                )
                messages.append({
                    "role": "tool",
                    "content": result.content[0].text,  # TextContent.text
                    "tool_call_id": tc["id"],
                })

    async def close(self) -> None:
        """Close all MCP connections (and terminate subprocesses)."""
        for stack in self._exit_stacks.values():
            await stack.aclose()

# Usage
async def main():
    agent = MCPAgent(llm_provider=OpenAIProvider(model="gpt-4o"))

    await agent.connect_server(
        "weather",
        StdioServerParameters(command="python", args=["weather_server.py"]),
    )
    await agent.connect_server(
        "filesystem",
        StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/docs"],
        ),
    )

    result = await agent.run("What's the weather in Tokyo and list my documents?")
    print(result)
    await agent.close()

asyncio.run(main())
```

> **Code Reference:** [Python](../../code/python/10-mcp-server/) · [Node.js](../../code/nodejs/10-mcp-server/) · [Go](../../code/go/10-mcp-server/)  
> The MCP folder includes a complete weather server, filesystem server, and MCP-enabled agent.

---

## MCP vs. Direct Function Calling

| | Direct Function Calling | MCP |
|:---|:---|:---|
| **Where tools live** | In your application code | In separate server processes |
| **Tool discovery** | You define tools at agent creation | Agent discovers tools at runtime |
| **Language coupling** | Tools must be in your app's language | Tools can be in any language |
| **Provider coupling** | OpenAI format, Anthropic format, etc. | MCP standard format |
| **Reusability** | Rewrite for each agent/app | One server, many clients |
| **Deployment** | Deploy with your app | Deploy independently |
| **Scaling** | Scales with your app | Scale servers independently |
| **Ecosystem** | None (ad-hoc) | Growing marketplace of pre-built servers |

### When to Use Direct Function Calling
- Simple tools with no reuse potential
- Tools that need tight coupling with application state
- Prototyping before extracting to MCP servers
- Tools that are trivial to implement

### When to Use MCP
- Tools used by multiple agents or applications
- Tools maintained by different teams
- Tools that need independent scaling
- When you want to use pre-built tools from the ecosystem
- When you're building a platform that third parties will extend

---

## The MCP Ecosystem

### Pre-Built MCP Servers

Anthropic and the community maintain a growing library:

| Server | What It Provides | Install |
|:---|:---|:---|
| **Filesystem** | Read, write, and search files | `npx @modelcontextprotocol/server-filesystem` |
| **GitHub** | Repository management, issues, PRs | `npx @modelcontextprotocol/server-github` |
| **Postgres** | Database querying with schema awareness | `npx @modelcontextprotocol/server-postgres` |
| **Slack** | Channel messaging and history | `npx @modelcontextprotocol/server-slack` |
| **Brave Search** | Web and local search | `npx @modelcontextprotocol/server-brave-search` |
| **Puppeteer** | Browser automation | `npx @modelcontextprotocol/server-puppeteer` |
| **Memory** | Persistent knowledge graph | `npx @modelcontextprotocol/server-memory` |
| **Git** | Repository operations | `npx @modelcontextprotocol/server-git` |

### Building Your Own

An MCP server is surprisingly simple. The minimum viable server:

```python
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("my-tool")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(
        name="hello",
        description="Say hello to someone",
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "hello":
        return [TextContent(type="text", text=f"Hello, {arguments['name']}!")]
    return [TextContent(type="text", text=f'{{"error": "Unknown tool: {name}}}')] 

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

asyncio.run(main())
```

That's it. 30 lines. Any agent can now say hello.

---

## Multi-Server Agent Architecture

A production agent might connect to multiple MCP servers:

```python
class MultiServerAgent:
    """
    Agent that dynamically connects to multiple MCP servers.
    """
    
    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.server_registry = ServerRegistry()
    
    async def setup(self, config: dict) -> None:
        """
        Connect to servers defined in configuration.
        
        config = {
            "servers": [
                {
                    "name": "weather",
                    "command": "python",
                    "args": ["servers/weather_server.py"],
                    "auto_connect": true
                },
                {
                    "name": "database",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-postgres", "$DATABASE_URL"],
                    "auto_connect": true
                }
            ]
        }
        """
        for server_config in config["servers"]:
            if server_config.get("auto_connect", True):
                await self.server_registry.connect(
                    server_config["name"],
                    StdioServerParameters(
                        command=server_config["command"],
                        args=server_config["args"]
                    )
                )
    
    async def run(self, user_input: str) -> str:
        """Run agent with all connected servers' tools."""
        all_tools = self.server_registry.get_all_tools()
        
        # Now the agent has tools from weather, database, filesystem, etc.
        # All discovered at runtime, all provider-agnostic
        ...

class ServerRegistry:
    """Manage multiple MCP server connections."""
    
    def __init__(self):
        self.servers: dict[str, ServerConnection] = {}
    
    async def connect(self, name: str, params) -> None:
        """Connect to a server and discover its tools."""
        ...
    
    def get_all_tools(self) -> list[dict]:
        """Get tools from all connected servers, namespaced by server name."""
        ...
    
    async def call_tool(self, server_name: str, tool_name: str, 
                       arguments: dict) -> str:
        """Call a tool on a specific server."""
        ...
    
    async def disconnect_all(self) -> None:
        """Close all server connections."""
        ...
```

---

## MCP and the Agent Lifecycle

MCP changes how agents are built. Instead of hardcoding tools, agents discover them:

```
START
  │
  ▼
┌─────────────────┐
│ Load Agent       │
│ Configuration    │  Which MCP servers to connect to
└────────┬────────┘
         ▼
┌─────────────────┐
│ Connect to MCP   │
│ Servers          │  Discover available tools at runtime
└────────┬────────┘
         ▼
┌─────────────────┐
│ Build Tool       │
│ Registry         │  Convert MCP schemas to LLM-compatible format
└────────┬────────┘
         ▼
┌─────────────────┐
│ Agent Loop       │  Now has tools from 3+ servers
│ (ReAct, etc.)    │  Provider-agnostic, language-agnostic
└────────┬────────┘
         ▼
┌─────────────────┐
│ Execute Tool     │  Routes to correct MCP server
│ via MCP          │  Server handles the actual execution
└────────┬────────┘
         ▼
┌─────────────────┐
│ Return Result    │  Server response → Agent → User
└─────────────────┘
```

---

## Common Pitfalls

- **"I raise exceptions from call_tool handlers"**: Raising an unhandled exception crashes the server process. Return an error payload instead: `return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]`. Clients can read this gracefully.
- **"I make every tool an MCP server"**: Not every tool needs to be a separate server process. A simple utility function can stay in your agent code. MCP is for tools that benefit from isolation, reuse, or independent deployment.
- **"I don't handle MCP server disconnections"**: MCP servers are separate processes. They can crash, hang, or disconnect. Wrap connections in `AsyncExitStack` (Python) or use the SDK's transport lifecycle hooks, and implement reconnection logic.
- **"I hardcode tool schemas in my agent AND in the MCP server"**: The server is the source of truth for tool schemas. The agent discovers them at runtime via `list_tools()`. Don't duplicate schema definitions; if you change the server schema, the client picks it up on the next connection.
- **"I use `stdio_client` as an awaitable instead of a context manager"**: `stdio_client(params)` is an async context manager, not a coroutine. Use `async with stdio_client(params) as (read, write):`.
- **"I assume all MCP servers use the same transport"**: Local servers use stdio; remote services use Streamable HTTP. Your client should support both. Most MCP SDKs handle this automatically via separate transport classes.
- **"I don't namespace tools from different servers"**: Two servers might both have a "search" tool. Always namespace by server name: `github__search`, `database__search`.
- **"I ignore MCP because my tools are simple today"**: MCP is infrastructure. It's easier to start with MCP than to migrate 50 direct function calls to MCP later. Start simple, but use the protocol.

## What's Next

You've completed the tool ecosystem section. You can now integrate any model, any vector database, observe every decision, and connect to any tool via MCP. Next: putting it all together with frameworks — when to use them and how to evaluate them.
→ [When to Use Frameworks](../06-frameworks-in-practice/01-when-to-use-frameworks.md)