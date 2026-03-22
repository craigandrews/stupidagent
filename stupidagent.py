import argparse
import os
import re
import subprocess
from typing import Dict, List, Tuple, NamedTuple

import requests
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
args = parser.parse_args()

DEBUG = args.debug


def debug_print(*args):
    if DEBUG:
        print(*args)


MODEL_NAME = os.getenv("MODEL_NAME", "stupid-agent")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
CONTEXT_WINDOW_TOKENS = int(os.getenv("CONTEXT_WINDOW_TOKENS", "4096"))


class SearchResult(NamedTuple):
    title: str
    url: str
    snippet: str


def read_file(path: str) -> str:
    path = path.lstrip('@')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"[Error reading {path}: {e}]"


def replace_file_references(text: str) -> str:
    pattern = r'@(\S+)'

    def replacer(match):
        path = match.group(1)
        debug_print("READING FILE:", path)
        content = read_file(path)
        return f"\n\n--- Content of {path} ---\n{content}\n--- End of {path} ---\n\n"

    return re.sub(pattern, replacer, text)


def count_tokens(text: str) -> int:
    return len(text) // 4


def compress_context(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(messages) <= 3:
        return messages

    recent_messages = messages[-3:]
    history = messages[:-3]

    summary_prompt = (
        "Summarize the following conversation history in a concise manner. "
        "Preserve key facts, decisions, and search queries. "
        "List any URLs mentioned in clear format.\n\n"
        "Conversation history:\n"
        + "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in history
        )
    )

    summary = call_ollama([{"role": "user", "content": summary_prompt}])

    return [{"role": "system", "content": f"Previous conversation summary:\n{summary}"}] + recent_messages


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


def call_ollama(messages: List[Dict[str, str]], model: str = MODEL_NAME) -> str:
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"].strip()


def parse_action(text: str):
    text = text.strip()

    if text.startswith("THINKING:"):
        thought = text[len("THINKING:"):].strip()
        if not thought:
            raise ValueError("Empty THINKING output")
        return ("thinking", thought)

    if text.startswith("SEARCH:"):
        query = text[len("SEARCH:"):].strip()
        if not query:
            raise ValueError("Empty SEARCH query")
        return ("search", query)

    if text.startswith("SHELL:"):
        command = text[len("SHELL:"):].strip()
        if not command:
            raise ValueError("Empty SHELL command")
        return ("shell", command)

    if text.startswith("READ:"):
        filename = text[len("READ:"):].strip()
        if not filename:
            raise ValueError("Empty READ filename")
        return ("read", filename)

    if text.startswith("FINAL:"):
        answer = text[len("FINAL:"):].strip()
        if not answer:
            raise ValueError("Empty FINAL answer")
        return ("final", answer)

    raise ValueError(f"Invalid model output: {text}")


def run_agent(user_input: str, messages: List[Dict[str, str]], max_steps: int = 5) -> Tuple[str, List[Dict[str, str]]]:
    user_input = replace_file_references(user_input)
    messages = messages.copy()
    messages.append({"role": "user", "content": user_input})

    for step in range(max_steps):
        debug_print(f"\n--- STEP {step + 1} ---")

        if step == max_steps - 1:
            messages.append({
                "role": "user",
                "content": "This is your final response. You MUST use FINAL: format."
            })

        model_output = call_ollama(messages)
        debug_print("MODEL OUTPUT:", model_output)

        try:
            action_type, payload = parse_action(model_output)
        except Exception as e:
            messages.append({"role": "assistant", "content": model_output})
            messages.append({
                "role": "user",
                "content": "Invalid format. Use a valid format.",
            })
            continue

        if action_type == "thinking":
            messages.append({"role": "assistant", "content": model_output})
            messages.append({"role": "user", "content": "continue"})
            print(f"... {payload}")
            continue

        if action_type == "final":
            messages.append({"role": "assistant", "content": model_output})
            return payload, messages

        if action_type == "search":
            try:
                results_text, results = web_search(payload)
            except Exception as e:
                results_text = f"Search failed: {e}"
                results = []

            debug_print("SEARCH QUERY:", payload)
            debug_print("SEARCH RESULTS:", results_text)

            link_context = ""
            if results:
                link_context = "\n\nRelevant links:\n" + "\n".join(
                    f"- {r.url}" for r in results
                )

            messages.append({"role": "assistant", "content": model_output})
            messages.append({
                "role": "user",
                "content": (
                    f"Search results:\n{results_text}\n\n"
                    "Answer directly. Include links if relevant. "
                    "Format: FINAL: <answer>"
                ) + link_context,
            })
            continue

        if action_type == "shell":
            try:
                result = subprocess.run(
                    ["bash", "-c", payload],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                output = result.stdout
                if result.stderr:
                    output += f"\n[STDERR]\n{result.stderr}"
                if result.returncode != 0:
                    output += f"\n[Exit code: {result.returncode}]"
            except Exception as e:
                output = f"Shell command failed: {e}"

            debug_print("SHELL COMMAND:", payload)
            debug_print("SHELL OUTPUT:", output)

            messages.append({"role": "assistant", "content": model_output})
            messages.append({
                "role": "user",
                "content": (
                    f"Shell command output:\n{output}\n\n"
                    "Answer directly based on the output. "
                    "Format: FINAL: <answer>"
                ),
            })
            continue

        if action_type == "read":
            content = read_file(payload)
            debug_print("READ FILE:", payload)
            debug_print("FILE CONTENT:", content)

            messages.append({"role": "assistant", "content": model_output})
            messages.append({
                "role": "user",
                "content": (
                    f"Content of {payload}:\n{content}\n\n"
                    "Answer directly based on the file content. "
                    "Format: FINAL: <answer>"
                ),
            })
            continue

    return "Agent stopped after reaching max_steps without producing a final answer.", messages


if __name__ == "__main__":
    messages = []
    try:
        while True:
            user_input = input(">>> ").strip()
            if user_input.lower() == "exit":
                break

            answer, messages = run_agent(user_input, messages)
            print(answer)

            total_tokens = sum(count_tokens(m["content"]) for m in messages)
            if total_tokens > CONTEXT_WINDOW_TOKENS:
                messages = compress_context(messages)
                debug_print("\n[Context compressed to maintain performance]")
    except EOFError:
        pass
    print("\nGoodbye!")
