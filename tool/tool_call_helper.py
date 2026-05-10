from __future__ import annotations
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional
import requests

DEFAULT_MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
DEFAULT_API_URL = os.getenv("ANTHROPIC_URL", "https://api.anthropic.com/v1/messages")
DEFAULT_ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_WAIT_SECONDS = float(os.getenv("WAIT_SECONDS", "3"))
DEFAULT_MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

def get_api_key(prompt = "Enter API key:"):
    if os.getenv("SKIP_API_KEY_PROMPT", "0") == "1":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key.strip():
            raise RuntimeError("SKIP_API_KEY_PROMPT=1 but ANTHROPIC_API_KEY is empty.")
        return key.strip().strip('"').strip("'")

    existing = os.getenv("ANTHROPIC_API_KEY", "")
    if existing.strip():
        return existing.strip().strip('"').strip("'")

    print(prompt)
    key = input().strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("Missing Anthropic API key.")
    os.environ["ANTHROPIC_API_KEY"] = key
    return key

def extract_json(raw_text):
    text = (raw_text or "").strip()
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

def build_headers(api_key, anthropic_version = DEFAULT_ANTHROPIC_VERSION):
    return {
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
        "content-type": "application/json",
    }

def _response_text(content_blocks):
    return "\n".join(
        block.get("text", "")
        for block in content_blocks
        if block.get("type") == "text"
    ).strip()

def _submit_only_tools(tools, submit_tool_name):
    filtered = [tool for tool in tools if tool.get("name") == submit_tool_name]
    if not filtered:
        raise RuntimeError(f"submit tool not found in tools: {submit_tool_name}")
    return filtered

def call_with_tool_use(
    *,
    api_key,
    user_prompt,
    tools: List[Dict[str, Any]],
    execute_tool: Callable[[str, Dict[str, Any]], Any],
    system_prompt = "",
    model_name = DEFAULT_MODEL_NAME,
    max_tokens = 3000,
    temperature = DEFAULT_TEMPERATURE,
    api_url = DEFAULT_API_URL,
    anthropic_version = DEFAULT_ANTHROPIC_VERSION,
    max_retries = DEFAULT_MAX_RETRIES,
    wait_seconds = DEFAULT_WAIT_SECONDS,
    submit_tool_name = "submit_answer",
    return_submitted_answer = True,
    continue_message = "Continue until complete. Use submit_answer for the final structured answer.",
):
    headers = build_headers(api_key, anthropic_version)
    last_error: Optional[Exception] = None
    last_text = ""

    if return_submitted_answer and not submit_tool_name:
        raise ValueError("return_submitted_answer=True requires submit_tool_name.")

    for attempt in range(1, max_retries + 1):
  
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        force_submit_next = False
        try:
            while True:
                current_tools = tools
                payload: Dict[str, Any] = {
                    "model": model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                }
                if system_prompt:
                    payload["system"] = system_prompt

                if tools:
                    if force_submit_next and submit_tool_name:
                        current_tools = _submit_only_tools(tools, submit_tool_name)
                        payload["tool_choice"] = {"type": "tool", "name": submit_tool_name}
                    payload["tools"] = current_tools

                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                last_text = response.text

                if response.status_code == 401:
                    raise RuntimeError(f"401 Unauthorized: {response.text}")
                if response.status_code == 429:
                    raise requests.HTTPError("429 Rate limit", response=response)

                response.raise_for_status()
                data = response.json()

                stop_reason = data.get("stop_reason")
                content_blocks = data.get("content", [])
                tool_uses = [block for block in content_blocks if block.get("type") == "tool_use"]

              
                if submit_tool_name:
                    for tool_use in tool_uses:
                        if tool_use.get("name") == submit_tool_name:
                            return json.dumps(tool_use.get("input", {}), ensure_ascii=False)

                messages.append({"role": "assistant", "content": content_blocks})

                if tool_uses:
                    tool_results = []
                    for tool_use in tool_uses:
                        tool_name = tool_use["name"]
                        tool_input = tool_use.get("input", {})
                        tool_use_id = tool_use["id"]
                        if submit_tool_name and tool_name == submit_tool_name:
                            return json.dumps(tool_input, ensure_ascii=False)

                        try:
                            result = execute_tool(tool_name, tool_input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps(result, ensure_ascii=False),
                            })
                        except Exception as exc:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "is_error": True,
                                "content": str(exc),
                            })

                    messages.append({"role": "user", "content": tool_results})
                    continue

                if stop_reason == "max_tokens":
                    
                    messages.append({"role": "user", "content": continue_message})
                    continue

                if stop_reason == "end_turn":
                    if return_submitted_answer and submit_tool_name:
                        force_submit_next = True
                        messages.append({
                            "role": "user",
                            "content": (
                                f"You must now call {submit_tool_name}. "
                                "Do not answer in free text. Return only the structured tool input."
                            ),
                        })
                        continue

                    return _response_text(content_blocks)

                messages.append({"role": "user", "content": continue_message})

        except requests.exceptions.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status in {400, 401, 403, 404}:
                raise
            sleep_for = wait_seconds * (2 ** (attempt - 1))
            print(f"HTTP error on attempt {attempt}/{max_retries}: {exc}. Sleeping {sleep_for:.1f}s.")
            time.sleep(sleep_for)

        except requests.exceptions.RequestException as exc:
            last_error = exc
            sleep_for = wait_seconds * (2 ** (attempt - 1))
            print(f"Request error on attempt {attempt}/{max_retries}: {exc}. Sleeping {sleep_for:.1f}s.")
            time.sleep(sleep_for)

        except RuntimeError as exc:
            last_error = exc
            if "401 Unauthorized" in str(exc) or "401 Unauthorized" in last_text:
                raise
            sleep_for = wait_seconds * (2 ** (attempt - 1))
            print(f"Runtime error on attempt {attempt}/{max_retries}: {exc}. Sleeping {sleep_for:.1f}s.")
            time.sleep(sleep_for)

    raise RuntimeError(f"Claude API call failed after {max_retries} retries: {last_error}")
