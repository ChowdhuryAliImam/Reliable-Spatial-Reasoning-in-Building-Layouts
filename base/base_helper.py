from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

@dataclass(frozen=True)
class LLMConfig:
    model_name: str
    n_runs: int
    wait_seconds: float
    max_retries: int
    output_dir: str
    temperature: float
    max_tokens: int
    anthropic_url: str = ANTHROPIC_URL
    anthropic_version: str = ANTHROPIC_VERSION

    @classmethod
    def from_env(cls, *, default_output_dir = "benchmark_outputs", default_n_runs = 5, default_wait_seconds = 3.0, default_max_retries = 5, default_temperature = 0.2, default_max_tokens = 700):
        return cls(
            model_name=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            n_runs=int(os.getenv("N_RUNS", str(default_n_runs))),
            wait_seconds=float(os.getenv("WAIT_SECONDS", str(default_wait_seconds))),
            max_retries=int(os.getenv("MAX_RETRIES", str(default_max_retries))),
            output_dir=os.getenv("OUTPUT_DIR", default_output_dir),
            temperature=float(os.getenv("TEMPERATURE", str(default_temperature))),
            max_tokens=int(os.getenv("MAX_TOKENS", str(default_max_tokens))),
            anthropic_version=os.getenv("ANTHROPIC_VERSION", ANTHROPIC_VERSION),
        )

def get_api_key():
    """Read Anthropic key from env or prompt once."""
    if os.getenv("SKIP_API_KEY_PROMPT", "0") == "1":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key.strip():
            raise RuntimeError("SKIP_API_KEY_PROMPT=1 but ANTHROPIC_API_KEY is empty.")
        return clean_api_key(key)

    existing = os.getenv("ANTHROPIC_API_KEY", "")
    if existing.strip():
        return clean_api_key(existing)

    print("Enter your Anthropic API key:")
    key = input().strip()
    if not key:
        raise RuntimeError("Missing Anthropic API key.")
    os.environ["ANTHROPIC_API_KEY"] = clean_api_key(key)
    return clean_api_key(key)

def clean_api_key(key):
    return key.strip().strip('"').strip("'")

def make_submit_answer_tool(input_schema):
   
    return [
        {
            "name": "submit_answer",
            "description": "Submit the final benchmark answer in the required structure.",
            "input_schema": input_schema,
        }
    ]

def extract_submit_answer_from_response(data):
    for block in data.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "submit_answer":
            return json.dumps(block.get("input", {}), ensure_ascii=False)
    return None

def call_claude(*, api_key, user_prompt, system_prompt, config, answer_schema = None):

    payload: Dict[str, Any] = {
        "model": config.model_name,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if system_prompt:
        payload["system"] = system_prompt
    if answer_schema is not None:
        payload["tools"] = make_submit_answer_tool(answer_schema)
        payload["tool_choice"] = {"type": "tool", "name": "submit_answer"}

    headers = {
        "x-api-key": api_key,
        "anthropic-version": config.anthropic_version,
        "content-type": "application/json",
    }

    last_error: Optional[Exception] = None
    last_text: str = ""

    for attempt in range(1, config.max_retries + 1):
        try:
            response = requests.post(
                config.anthropic_url,
                headers=headers,
                json=payload,
                timeout=60,
            )
            last_text = response.text

            if response.status_code == 401:
                raise RuntimeError(
                    "Claude API returned 401 Unauthorized. Check API key and model access. "
                    f"Response text: {last_text}"
                )

            if response.status_code == 429:
                raise requests.HTTPError("429 Rate limit", response=response)

            response.raise_for_status()
            data = response.json()

            if answer_schema is not None:
                structured = extract_submit_answer_from_response(data)
                if structured is not None:
                    return structured
                raise RuntimeError(f"Claude did not call submit_answer. Response text: {last_text}")

            return extract_text_from_response(data)

        except requests.exceptions.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status in {400, 401, 403, 404}:
                raise
            sleep_with_backoff(attempt, config.wait_seconds, f"HTTP error on attempt {attempt}/{config.max_retries}: {exc}")

        except requests.exceptions.RequestException as exc:
            last_error = exc
            sleep_with_backoff(attempt, config.wait_seconds, f"Request error on attempt {attempt}/{config.max_retries}: {exc}")

        except RuntimeError as exc:
            last_error = exc
            if "401 Unauthorized" in str(exc):
                raise
            sleep_with_backoff(attempt, config.wait_seconds, f"Runtime error on attempt {attempt}/{config.max_retries}: {exc}")

    raise RuntimeError(f"Claude API call failed after {config.max_retries} retries: {last_error}")

def extract_text_from_response(data):
    blocks = data.get("content", [])
    texts = [block.get("text", "") for block in blocks if block.get("type") == "text"]
    return "\n".join(texts).strip()

def sleep_with_backoff(attempt, wait_seconds, message):
    sleep_for = wait_seconds * (2 ** (attempt - 1))
    print(f"{message}. Sleeping {sleep_for:.1f}s.")
    time.sleep(sleep_for)

def pause_between_runs(wait_seconds):
    if wait_seconds > 0:
        time.sleep(wait_seconds)

def extract_json(raw_text):
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None

def extract_xml_tag(tag, text):
    if not text:
        return None
    try:
        return re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL).group(1).strip()
    except Exception:
        return None

def sorted_unique(items):
    return sorted(set(items))

def graph_to_json(adj):
    normalized = {node: sorted_unique(neighbors) for node, neighbors in sorted(adj.items())}
    return json.dumps(normalized, indent=2)

def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)

def rank_top_k(values, *, k = 3, mode = "lowest"):
    if mode == "lowest":
        ranked = sorted(values.items(), key=lambda item: (float(item[1]), str(item[0])))
    elif mode == "highest":
        ranked = sorted(values.items(), key=lambda item: (-float(item[1]), str(item[0])))
    else:
        raise ValueError("mode must be 'lowest' or 'highest'")
    return [str(node) for node, _ in ranked[:k]]

def top3_accuracy(predicted, expected):
    if predicted is None:
        return 0.0, 0, len(expected), []
    correct_positions = [
        f"pos_{idx + 1}:{gt}"
        for idx, (pred, gt) in enumerate(zip(predicted, expected))
        if pred == gt
    ]
    total = len(expected)
    num_correct = len(correct_positions)
    return (num_correct / total if total else 0.0), num_correct, total, correct_positions

def consistency_score(values):
    if not values:
        return 0.0
    serialized = [json.dumps(v, sort_keys=True, ensure_ascii=False) for v in values]
    counts: Dict[str, int] = {}
    for item in serialized:
        counts[item] = counts.get(item, 0) + 1
    return max(counts.values()) / len(values)

def binary_consistency(values):
    if not values:
        return 0.0
    ones = sum(int(v) for v in values)
    zeros = len(values) - ones
    return max(ones, zeros) / len(values)

def pass_at_k(scores):
    return int(any(scores))

def pass_all(scores):
    return int(all(scores))
