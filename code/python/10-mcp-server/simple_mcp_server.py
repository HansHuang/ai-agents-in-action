"""SimpleMCPServer — build MCP servers with minimal boilerplate.

Usage::

    server = SimpleMCPServer("my-tools")

    @server.tool()
    def add(a: int, b: int) -> int:
        '''Add two numbers together.'''
        return a + b

    @server.resource("config://settings")
    def get_settings() -> str:
        return json.dumps({"version": "1.0"})

    server.run()   # listens on stdio

Run the demo::

    python simple_mcp_server.py
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
import types
from typing import Any, Callable, get_args, get_origin, get_type_hints

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import AnyUrl, Resource, TextContent, Tool


# ---------------------------------------------------------------------------
# Type → JSON Schema helpers
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[type, str] = {
    str:   "string",
    int:   "integer",
    float: "number",
    bool:  "boolean",
    list:  "array",
    dict:  "object",
}


def _annotation_to_schema(annotation: Any) -> dict:
    """Convert one Python type annotation to a JSON Schema fragment."""
    origin = get_origin(annotation)

    # Optional[X]  (i.e. Union[X, None])
    if origin is types.UnionType:
        # Python 3.10+ ``X | None`` syntax
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _annotation_to_schema(args[0])
        return {"type": "string"}  # fallback for complex unions

    try:
        import typing
        if origin is typing.Union:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                return _annotation_to_schema(args[0])
    except AttributeError:
        pass

    # list[X]
    if origin is list:
        schema: dict = {"type": "array"}
        item_args = get_args(annotation)
        if item_args:
            schema["items"] = _annotation_to_schema(item_args[0])
        return schema

    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}

    return {"type": "string"}  # safe fallback


def _generate_schema(func: Callable) -> dict:
    """Generate an MCP ``inputSchema`` from a function's type hints.

    Rules:
    - Each parameter → a JSON Schema property.
    - Type hints determine the JSON Schema type (str→string, int→integer, …).
    - Parameters without defaults are ``required``.
    - ``self`` and ``cls`` are ignored.
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = hints.get(param_name, str)
        prop = _annotation_to_schema(annotation)
        prop["description"] = f"The '{param_name}' parameter."
        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _get_description(func: Callable, override: str | None) -> str:
    """Return the description from *override* or the first line of the docstring."""
    if override:
        return override
    doc = inspect.getdoc(func)
    if doc:
        return doc.splitlines()[0].strip()
    return func.__name__


# ---------------------------------------------------------------------------
# SimpleMCPServer
# ---------------------------------------------------------------------------

class SimpleMCPServer:
    """Build MCP servers with minimal boilerplate using decorators.

    Example::

        server = SimpleMCPServer("my-tools")

        @server.tool()
        def hello(name: str) -> str:
            '''Say hello to someone.'''
            return f"Hello, {name}!"

        @server.tool()
        def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b

        @server.resource("config://version")
        def version() -> str:
            return '{"version": "1.0.0"}'

        server.run()
    """

    def __init__(self, name: str, version: str = "1.0.0") -> None:
        self.name = name
        self.version = version
        self._server = Server(name)
        self._tools: dict[str, tuple[Tool, Callable]] = {}
        self._resources: dict[str, tuple[Resource, Callable]] = {}
        self._registered = False

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable:
        """Decorator that registers a function as an MCP tool.

        ``inputSchema`` is auto-generated from type hints.
        ``description`` defaults to the first line of the docstring.

        Args:
            name:        Override for the tool name (default: function name).
            description: Override for the tool description.
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = _get_description(func, description)
            schema = _generate_schema(func)
            mcp_tool = Tool(
                name=tool_name,
                description=tool_desc,
                inputSchema=schema,
            )
            self._tools[tool_name] = (mcp_tool, func)
            print(
                f"[{self.name}] Tool registered: {tool_name!r} "
                f"({len(schema.get('required', []))} required params)",
                file=sys.stderr,
            )
            return func

        return decorator

    def resource(
        self,
        uri: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str = "application/json",
    ) -> Callable:
        """Decorator that registers a function as an MCP resource.

        The function is called when a client reads the resource and
        should return a string (JSON, plain text, etc.).
        """
        def decorator(func: Callable) -> Callable:
            resource_name = name or func.__name__
            resource_desc = _get_description(func, description)
            mcp_resource = Resource(
                uri=AnyUrl(uri),
                name=resource_name,
                description=resource_desc,
                mimeType=mime_type,
            )
            self._resources[uri] = (mcp_resource, func)
            print(
                f"[{self.name}] Resource registered: {uri!r}",
                file=sys.stderr,
            )
            return func

        return decorator

    # ------------------------------------------------------------------
    # Internal: register MCP handlers exactly once
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        if self._registered:
            return
        self._registered = True

        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            return [t for t, _ in self._tools.values()]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            entry = self._tools.get(name)
            if entry is None:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Unknown tool: {name!r}"}
                ))]
            _, func = entry
            try:
                result = func(**arguments)
                if asyncio.iscoroutine(result):
                    result = await result
                text = result if isinstance(result, str) else json.dumps(result)
                return [TextContent(type="text", text=text)]
            except TypeError as exc:
                return [TextContent(type="text", text=json.dumps(
                    {"error": f"Invalid arguments: {exc}"}
                ))]
            except Exception as exc:  # noqa: BLE001
                print(f"[{self.name}] Error in tool '{name}': {exc}", file=sys.stderr)
                return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

        if self._resources:
            @self._server.list_resources()
            async def list_resources() -> list[Resource]:
                return [r for r, _ in self._resources.values()]

            @self._server.read_resource()
            async def read_resource(uri: AnyUrl) -> str:
                uri_str = str(uri)
                entry = self._resources.get(uri_str)
                if entry is None:
                    raise ValueError(f"Unknown resource URI: {uri_str!r}")
                _, func = entry
                result = func()
                if asyncio.iscoroutine(result):
                    result = await result
                return result if isinstance(result, str) else json.dumps(result)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def show_schemas(self) -> None:
        """Print all registered tool schemas to stderr (for inspection)."""
        print(f"\n[{self.name}] Registered tools:", file=sys.stderr)
        for tool_name, (tool, _) in self._tools.items():
            print(
                f"\n  {tool_name}:\n"
                + json.dumps(tool.inputSchema, indent=4),
                file=sys.stderr,
            )

    def run(self, transport: str = "stdio") -> None:
        """Start the server.  Only stdio transport is currently supported."""
        if transport != "stdio":
            raise ValueError(
                f"Unsupported transport: {transport!r}. Only 'stdio' is supported."
            )
        self._register_handlers()

        async def _run() -> None:
            print(f"[{self.name}] Starting ({transport}) ...", file=sys.stderr)
            async with stdio_server() as (read, write):
                await self._server.run(
                    read,
                    write,
                    self._server.create_initialization_options(),
                )

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Demo: 4-tool server
# ---------------------------------------------------------------------------

def _build_demo_server() -> SimpleMCPServer:
    """Build a demo server with four tools and one resource."""
    server = SimpleMCPServer("demo-tools", version="1.0.0")

    @server.tool()
    def hello(name: str) -> str:
        """Say hello to someone by name."""
        return f"Hello, {name}!"

    @server.tool()
    def calculate(expression: str) -> str:
        """Evaluate a safe mathematical expression.

        Supports +, -, *, /, **, and parentheses.
        Example: '(2 + 3) * 4'
        """
        allowed = set("0123456789 +-*/().**e ")
        if not all(c in allowed for c in expression.replace(" ", "")):
            return json.dumps({"error": "Expression contains unsafe characters."})
        try:
            result = eval(expression, {"__builtins__": {}})  # noqa: S307
            return json.dumps({"expression": expression, "result": result})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @server.tool()
    def search_files(directory: str, pattern: str) -> str:
        """Search for files matching a glob pattern in a directory.

        Returns a list of matching file paths.
        """
        import glob
        import os

        full_pattern = os.path.join(directory, "**", pattern)
        matches = glob.glob(full_pattern, recursive=True)
        return json.dumps({
            "directory": directory,
            "pattern": pattern,
            "matches": matches,
            "count": len(matches),
        })

    @server.tool()
    def send_notification(message: str, channel: str) -> str:
        """Send a notification to a channel.

        Supported channels: 'slack', 'email', 'log'.
        Replace this stub with a real integration in production.
        """
        # In production: call Slack API, SES, PagerDuty, etc.
        print(f"[NOTIFICATION → {channel}] {message}", file=sys.stderr)
        return json.dumps({"status": "sent", "channel": channel, "message": message})

    @server.resource("config://version")
    def get_version() -> str:
        """Return server version information."""
        return json.dumps({"name": "demo-tools", "version": "1.0.0"})

    return server


if __name__ == "__main__":
    demo_server = _build_demo_server()
    demo_server.show_schemas()  # Print schemas to stderr before starting
    demo_server.run()
