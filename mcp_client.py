import json
from contextlib import AsyncExitStack
from typing import Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client


class ToolInfo:
    def __init__(
        self, server_name: str, name: str, description: str, input_schema: dict
    ):
        self.server_name = server_name
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def qualified_name(self) -> str:
        return f"{self.server_name}.{self.name}"

    def format_for_prompt(self) -> str:
        params = self.input_schema.get("properties", {})
        param_parts = []
        required = set(self.input_schema.get("required", []))
        for pname, pschema in params.items():
            ptype = pschema.get("type", "any")
            opt = "" if pname in required else ", optional"
            param_parts.append(f"{pname}: {ptype}{opt}")
        params_str = ", ".join(param_parts) if param_parts else "none"
        return f"- {self.qualified_name()}: {self.description} (params: {params_str})"


class MCPManager:
    def __init__(self):
        self._exit_stack = AsyncExitStack()
        self._sessions: Dict[str, ClientSession] = {}
        self._tools: List[ToolInfo] = []

    async def connect_all(self, config: dict, debug_print=None):
        servers = config.get("mcpServers", {})
        for name, server_config in servers.items():
            try:
                await self._connect_server(name, server_config)
                if debug_print:
                    debug_print(f"Connected to MCP server: {name}")
            except Exception as e:
                print(f"Failed to connect to MCP server '{name}': {e}")

    async def _connect_server(self, name: str, config: dict):
        if "command" in config:
            params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env"),
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
        elif "url" in config:
            url = config["url"]
            read, write = await self._exit_stack.enter_async_context(
                sse_client(url=url)
            )
        else:
            raise ValueError(f"Server '{name}' must have either 'command' or 'url'")

        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[name] = session

        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            self._tools.append(
                ToolInfo(
                    server_name=name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema
                    if isinstance(tool.inputSchema, dict)
                    else {},
                )
            )

    def list_all_tools(self) -> List[ToolInfo]:
        return list(self._tools)

    def has_tools(self) -> bool:
        return len(self._tools) > 0

    def tools_prompt_section(self) -> str:
        if not self._tools:
            return ""
        lines = ["You have access to the following tools:"]
        for tool in self._tools:
            lines.append(tool.format_for_prompt())
        lines.append("")
        lines.append(
            'Use TOOL: server_name.tool_name {"param": "value"} to invoke them.'
        )
        return "\n".join(lines)

    async def call_tool(self, qualified_name: str, arguments: dict) -> str:
        parts = qualified_name.split(".", 1)
        if len(parts) != 2:
            return f"Invalid tool name '{qualified_name}'. Use server_name.tool_name format."

        server_name, tool_name = parts
        session = self._sessions.get(server_name)
        if not session:
            return f"No connected server named '{server_name}'. Available: {', '.join(self._sessions.keys())}"

        result = await session.call_tool(tool_name, arguments)
        text_parts = []
        for content in result.content:
            if hasattr(content, "text"):
                text_parts.append(content.text)
            else:
                text_parts.append(str(content))
        return "\n".join(text_parts) if text_parts else "(no output)"

    async def close_all(self):
        await self._exit_stack.aclose()


def load_mcp_config(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading MCP config from {path}: {e}")
        return None
