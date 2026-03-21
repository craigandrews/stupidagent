# Stupid Agent

A simple web-searching agent powered by Ollama and the Brave Search API.

## What It Does

The agent engages in conversation, performs web searches when needed, and provides answers with source links. It operates in an iterative loop:

1. Receives user input
2. Decides whether to search or answer
3. If searching: queries Brave Search, incorporates results
4. Final response includes relevant links

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

Or specify a custom model:

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
- **Automatic context compression**: Trims old messages when context window fills
- **File reading**: Prefix paths with `@` to include file contents (e.g., `@readme.md`)
- **Link-aware**: Returns source URLs in answers

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BRAVE_API_KEY` | (required) | Your Brave Search API key |
| `CONTEXT_WINDOW_TOKENS` | 4096 | Max tokens before compression |
| `MODEL_NAME` | `simple-agent` | Ollama model name to use |

**Note:** The `MODEL_NAME` in `stupidagent.py` should match the model name used when creating the Ollama model (e.g., `stupid-agent` from `ollama create`).

## How It Works

- Uses Ollama's `/api/chat` endpoint with a simple prompt
- Parses responses for `SEARCH:` or `FINAL:` actions
- Searches via Brave's Web Search API
- Compresses conversation history by summarizing old messages when context exceeds the token limit
