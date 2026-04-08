import argparse
import asyncio
import json
import os
import re
import readline  # noqa: F401  # activates line editing for input()
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, NamedTuple

import requests
from dotenv import load_dotenv

from mcp_client import MCPManager, load_mcp_config

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug", action="store_true", help="Enable debug mode with verbose output"
)
parser.add_argument(
    "--mcp-config", default="mcp_servers.json", help="Path to MCP servers config file"
)
args = parser.parse_args()

DEBUG = args.debug


def debug_print(*a):
    if DEBUG:
        print(*a)


MODEL_NAME = os.getenv("MODEL_NAME", "stupid-agent")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
CONTEXT_WINDOW_TOKENS = int(os.getenv("CONTEXT_WINDOW_TOKENS", "4096"))


class SearchResult(NamedTuple):
    title: str
    url: str
    snippet: str


def read_file(path: str) -> str:
    path = path.lstrip("@")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[Error reading {path}: {e}]"


def replace_file_references(text: str) -> str:
    pattern = r"@(\S+)"

    def replacer(match):
        path = match.group(1)
        debug_print("READING FILE:", path)
        content = read_file(path)
        return f"\n\n--- Content of {path} ---\n{content}\n--- End of {path} ---\n\n"

    return re.sub(pattern, replacer, text)


def count_tokens(text: str) -> int:
    return len(text) // 4


async def compress_context(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(messages) <= 3:
        return messages

    recent_messages = messages[-3:]
    history = messages[:-3]

    summary_prompt = (
        "Summarize the following conversation history in a concise manner. "
        "Preserve key facts, decisions, and search queries. "
        "List any URLs mentioned in clear format.\n\n"
        "Conversation history:\n"
        + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    )

    summary = await call_ollama([{"role": "user", "content": summary_prompt}])

    return [
        {"role": "system", "content": f"Previous conversation summary:\n{summary}"}
    ] + recent_messages


def extract_links(text: str) -> List[str]:
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
    return re.findall(url_pattern, text)


def web_search(query: str, count: int = 5) -> Tuple[str, List[SearchResult]]:
    if not BRAVE_API_KEY:
        raise ValueError("BRAVE_API_KEY not set")

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "count": count,
    }

    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    results_data = data.get("web", {}).get("results", [])
    if not results_data:
        return f"No results found for query: {query}", []

    results: List[SearchResult] = []
    lines = []
    for i, r in enumerate(results_data[:count], start=1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        desc = (r.get("description") or "").strip()
        if len(desc) > 300:
            desc = desc[:300] + "..."
        lines.append(f"{i}. {title}\nURL: {url}\nSnippet: {desc}")
        results.append(SearchResult(title=title, url=url, snippet=desc))

    return "\n\n".join(lines), results


async def call_ollama(messages: List[Dict[str, str]], model: str = MODEL_NAME) -> str:
    def _post():
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": True,
                "options": {
                    "temperature": 1,
                    "top_p": 0.95,
                    "top_k": 64
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    data = await asyncio.to_thread(_post)
    if "thinking" in data["message"]:
        print("--think--", data["message"]["thinking"].replace("\n", " "))
    return data["message"]["content"].strip()


@dataclass
class Action:
    type: str
    payload: str
    arguments: Dict[str, object] = field(default_factory=dict)


def parse_action(text: str) -> Action:
    text = text.strip()

    if text.startswith("THINKING:"):
        thought = text[len("THINKING:") :].strip()
        if not thought:
            raise ValueError("Empty THINKING output")
        return Action("thinking", thought)

    if text.startswith("SEARCH:"):
        query = text[len("SEARCH:") :].strip()
        if not query:
            raise ValueError("Empty SEARCH query")
        return Action("search", query)

    if text.startswith("SHELL:"):
        command = text[len("SHELL:") :].strip()
        if not command:
            raise ValueError("Empty SHELL command")
        return Action("shell", command)

    if text.startswith("READ:"):
        filename = text[len("READ:") :].strip()
        if not filename:
            raise ValueError("Empty READ filename")
        return Action("read", filename)

    if text.startswith("TOOL:"):
        rest = text[len("TOOL:") :].strip()
        if not rest:
            raise ValueError("Empty TOOL invocation")
        # Format: server.tool_name {"arg": "value"}
        # Find the JSON object start
        json_start = rest.find("{")
        if json_start == -1:
            return Action("tool", rest.strip())
        tool_name = rest[:json_start].strip()
        try:
            arguments = json.loads(rest[json_start:])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid TOOL JSON arguments: {e}")
        return Action("tool", tool_name, arguments)

    if text.startswith("FINAL:"):
        answer = text[len("FINAL:") :].strip()
        if not answer:
            raise ValueError("Empty FINAL answer")
        return Action("final", answer)

    raise ValueError(f"Invalid model output: {text}")


async def run_agent(
    user_input: str,
    messages: List[Dict[str, str]],
    mcp: Optional[MCPManager] = None,
    max_steps: int = 5,
) -> Tuple[str, List[Dict[str, str]]]:
    user_input = replace_file_references(user_input)
    messages = messages.copy()
    messages.append({"role": "user", "content": user_input})

    for step in range(max_steps):
        debug_print(f"\n--- STEP {step + 1} ---")

        if step == max_steps - 1:
            messages.append(
                {
                    "role": "user",
                    "content": "This is your final response. You MUST use FINAL: format.",
                }
            )

        model_output = await call_ollama(messages)
        debug_print("MODEL OUTPUT:", model_output)

        try:
            parsed = parse_action(model_output)
        except Exception as e:
            messages.append({"role": "assistant", "content": model_output})
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid format. Use a valid format.",
                }
            )
            continue

        action = parsed

        if action.type == "thinking":
            messages.append({"role": "assistant", "content": model_output})
            messages.append({"role": "user", "content": "continue"})
            print(f"... {action.payload}")
            continue

        if action.type == "final":
            messages.append({"role": "assistant", "content": model_output})
            return action.payload, messages

        if action.type == "search":
            try:
                results_text, results = web_search(action.payload)
            except Exception as e:
                results_text = f"Search failed: {e}"
                results = []

            debug_print("SEARCH QUERY:", action.payload)
            debug_print("SEARCH RESULTS:", results_text)

            link_context = ""
            if results:
                link_context = "\n\nRelevant links:\n" + "\n".join(
                    f"- {r.url}" for r in results
                )

            messages.append({"role": "assistant", "content": model_output})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Search results:\n{results_text}\n\n"
                        "Answer directly. Include links if relevant. "
                        "Format: FINAL: <answer>"
                    )
                    + link_context,
                }
            )
            continue

        if action.type == "shell":
            try:
                result = subprocess.run(
                    ["bash", "-c", action.payload],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[STDERR]\n{result.stderr}"
                if result.returncode != 0:
                    output += f"\n[Exit code: {result.returncode}]"
            except Exception as e:
                output = f"Shell command failed: {e}"

            debug_print("SHELL COMMAND:", action.payload)
            debug_print("SHELL OUTPUT:", output)

            messages.append({"role": "assistant", "content": model_output})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Shell command output:\n{output}\n\n"
                        "Answer directly based on the output. "
                        "Format: FINAL: <answer>"
                    ),
                }
            )
            continue

        if action.type == "read":
            content = read_file(action.payload)
            debug_print("READ FILE:", action.payload)
            debug_print("FILE CONTENT:", content)

            messages.append({"role": "assistant", "content": model_output})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Content of {action.payload}:\n{content}\n\n"
                        "Answer directly based on the file content. "
                        "Format: FINAL: <answer>"
                    ),
                }
            )
            continue

        if action.type == "tool":
            if not mcp:
                output = "No MCP servers configured."
            else:
                debug_print("TOOL CALL:", action.payload, action.arguments)
                try:
                    output = await mcp.call_tool(action.payload, action.arguments)
                except Exception as e:
                    output = f"Tool call failed: {e}"
                debug_print("TOOL OUTPUT:", output)

            messages.append({"role": "assistant", "content": model_output})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool output:\n{output}\n\n"
                        "Answer directly based on the tool output. "
                        "Format: FINAL: <answer>"
                    ),
                }
            )
            continue

    return (
        "Agent stopped after reaching max_steps without producing a final answer.",
        messages,
    )


async def main():
    mcp = MCPManager()
    config = load_mcp_config(args.mcp_config)
    if config:
        await mcp.connect_all(config, debug_print=debug_print)
        tools_section = mcp.tools_prompt_section()
        if tools_section:
            print(f"MCP tools available: {len(mcp.list_all_tools())}")
            debug_print(tools_section)
    else:
        tools_section = ""

    # Inject tool descriptions into system message if any MCP tools are connected
    messages: List[Dict[str, str]] = []
    if tools_section:
        messages.append({"role": "system", "content": tools_section})

    try:
        while True:
            user_input = (await asyncio.to_thread(input, ">>> ")).strip()
            if user_input.lower() == "exit":
                break

            answer, messages = await run_agent(user_input, messages, mcp=mcp)
            debug_print("\nResponse:")
            print(answer)

            total_tokens = sum(count_tokens(m["content"]) for m in messages)
            if total_tokens > CONTEXT_WINDOW_TOKENS:
                messages = await compress_context(messages)
                debug_print("\n[Context compressed to maintain performance]")
    except EOFError:
        pass
    finally:
        await mcp.close_all()

    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
