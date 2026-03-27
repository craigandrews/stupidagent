# Stupid Agent

A simple agentic loop powered by Ollama and the Brave Search API.

It's really stupid.

## What It Does

The agent engages in conversation and can use several actions to answer questions. It operates in an iterative loop (up to 5 steps per turn):

1. Receives user input
2. Decides which action to take: think, search the web, run a shell command, read a file, call an MCP tool, or produce a final answer
3. Incorporates action results and repeats until it produces a `FINAL:` response

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- [Ollama](https://ollama.com/) running locally
- Brave Search API key

## Setup

1. Clone and navigate to the directory

2. Create `.env` file:
   ```env
   BRAVE_API_KEY=your_brave_api_key_here
   ```

3. Build the Ollama model:
   ```bash
   ollama create stupid-agent -f Modelfile
   ```

4. Install project dependencies:
   ```bash
   uv sync
   ```

4. Start Ollama (if not already running):
   ```bash
   ollama serve
   ```

## Usage

```bash
uv run python stupidagent.py
```

Enable debug output:

```bash
uv run python stupidagent.py --debug
```

Use a custom MCP server config:

```bash
uv run python stupidagent.py --mcp-config path/to/config.json
```

Specify a custom model:

```bash
MODEL_NAME=stupid-agent uv run python stupidagent.py
```

Then chat interactively:

```
User: What is the latest Python version?
Agent: [performs search and responds with answer and links]
```

Type `exit` or press Ctrl+D to quit.

## Features

- **Conversational**: Maintains context across multiple turns
- **Automatic context compression**: Summarizes old messages when the context window fills
- **Web search**: Searches the web via the Brave Search API (`SEARCH:` action)
- **Shell commands**: Runs arbitrary bash commands with a 30-second timeout (`SHELL:` action)
- **File reading**: Reads local files (`READ:` action), or inline file contents in prompts by prefixing paths with `@` (e.g., `@readme.md`)
- **Thinking**: Intermediate reasoning steps before acting (`THINKING:` action)
- **MCP tool support**: Connects to external tool servers via the [Model Context Protocol](https://modelcontextprotocol.io/) (`TOOL:` action). Configure servers in `mcp_servers.json` (or a custom path via `--mcp-config`). Supports both stdio and SSE transports.
- **Link-aware**: Returns source URLs in answers

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAVE_API_KEY` | (required) | Your Brave Search API key |
| `CONTEXT_WINDOW_TOKENS` | 4096 | Max tokens before compression |
| `MODEL_NAME` | `stupid-agent` | Ollama model name to use |

**Note:** The `MODEL_NAME` in `stupidagent.py` should match the model name used when creating the Ollama model (e.g., `stupid-agent` from `ollama create`).

## How It Works

- Uses Ollama's `/api/chat` endpoint with temperature 0
- Parses each response for a single action prefix: `THINKING:`, `SEARCH:`, `SHELL:`, `READ:`, `TOOL:`, or `FINAL:`
- Searches via Brave's Web Search API
- Runs shell commands via `bash -c` with a 30-second timeout
- Reads local files and injects their contents into the conversation
- Calls MCP tools on connected servers (stdio or SSE)
- Compresses conversation history by summarizing old messages when context exceeds the token limit
