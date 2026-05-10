from __future__ import annotations

"""Global/metric graph-reasoning benchmark for Claude using metric tools + semantic KG tools.

Tasks:
- T4 mean depth values for all spaces + top 3 lowest mean depth
- T5 unnormalized choice values for all spaces + top 3 highest choice

This is an experiment script, not only an export script. It calls Claude with tool_use,
allows metric tools and semantic KG query tools, and forces final structured output
through the submit_answer tool.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

import connectivity_graph_data
import Tools
from export_utils import export_csv
from kg_build import (
    Triple,
    build_semantic_kg_triples,
    get_circulation_spaces_from_kg,
    get_entries_from_kg,
    get_private_spaces_from_kg,
    get_shared_or_guest_facing_spaces_from_kg,
    tails_for_head,
)

# Config

MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
N_RUNS = int(os.getenv("N_RUNS", "5"))
WAIT_SECONDS = float(os.getenv("WAIT_SECONDS", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
API_URL = os.getenv("ANTHROPIC_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1400"))
MODULE_NAME = "global_kg"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", SCRIPT_DIR)
REL_TOLERANCE = float(os.getenv("REL_TOLERANCE", "0.05"))
ABS_TOLERANCE_FOR_ZERO = float(os.getenv("ABS_TOLERANCE_FOR_ZERO", "1e-6"))
ROUND_VALUES_IN_GT = int(os.getenv("ROUND_VALUES_IN_GT", "4"))

graphs: Dict[str, Dict[str, List[str]]] = {
    "graph_1": connectivity_graph_data.graph_1,
    "graph_2": connectivity_graph_data.graph_2,
    "graph_3": connectivity_graph_data.graph_3,
    "graph_4": connectivity_graph_data.graph_4,
    "graph_5": connectivity_graph_data.graph_5,
}

@dataclass
class MetricTask:
    task_id: str
    task_number: int
    task_type: str
    graph_name: str
    question: str
    expected_metric_values: Dict[str, float]
    expected_top_3: List[str]
    top_3_key: str
    ranking_rule: str
    metadata: Dict[str, Any]

# Api key

def get_api_key():
    if os.getenv("SKIP_API_KEY_PROMPT", "0") == "1":
        key = os.getenv("ANTHROPIC_API_KEY", "")
    else:
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key.strip():
            print("Enter your Anthropic API key:")
            key = input().strip()

    key = key.strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("Missing Anthropic API key.")
    os.environ["ANTHROPIC_API_KEY"] = key
    return key

# Ground-truth utilities

def sorted_unique(items):
    return sorted(set(items))

def graph_to_json(adjacency):
    normalized = {node: sorted_unique(neighbors) for node, neighbors in sorted(adjacency.items())}
    return json.dumps(normalized, indent=2)

def kg_triples_to_json(triples):
    rows = [{"head": h, "relation": r, "tail": t} for h, r, t in triples]
    return json.dumps(rows, indent=2, ensure_ascii=False)

def round_metric_dict(values, digits = ROUND_VALUES_IN_GT):
    return {str(k): round(float(v), digits) for k, v in sorted(values.items())}

def rank_top_3(values, mode):
    if mode == "lowest":
        ranked = sorted(values.items(), key=lambda item: (float(item[1]), str(item[0])))
    elif mode == "highest":
        ranked = sorted(values.items(), key=lambda item: (-float(item[1]), str(item[0])))
    else:
        raise ValueError("mode must be 'lowest' or 'highest'")
    return [str(node) for node, _ in ranked[:3]]

def calculate_choice(adjacency, normalized = False):
    try:
        return Tools.calculate_choice(adjacency, normalized=normalized)
    except TypeError:
        return Tools.calculate_choice(adjacency)

def make_tasks():
    tasks: List[MetricTask] = []
    for graph_name, adjacency in graphs.items():
        mean_depth_gt = round_metric_dict(Tools.calculate_mean_depth(adjacency))
        tasks.append(MetricTask(
            task_id=f"{graph_name}_T4_mean_depth",
            task_number=4,
            task_type="T4_mean_depth",
            graph_name=graph_name,
            question=(
                "For the given layout graph, compute the mean depth of each space. "
                "Return two things: (1) the mean depth value for every space, and "
                "(2) the top three spaces with the lowest mean depth."
            ),
            expected_metric_values=mean_depth_gt,
            expected_top_3=rank_top_3(mean_depth_gt, mode="lowest"),
            top_3_key="top_3_lowest_mean_depth",
            ranking_rule="Lower mean depth ranks higher. Ties are broken alphabetically.",
            metadata={"metric": "mean_depth", "ranking": "lowest", "tolerance": REL_TOLERANCE},
        ))

        choice_gt = round_metric_dict(calculate_choice(adjacency, normalized=False))
        tasks.append(MetricTask(
            task_id=f"{graph_name}_T5_choice",
            task_number=5,
            task_type="T5_choice",
            graph_name=graph_name,
            question=(
                "For the given layout graph, compute the unnormalized choice value of each space. "
                "Choice is defined as the number of shortest paths between all pairs of spaces "
                "that pass through that space. Return two things: (1) the choice value for every space, "
                "and (2) the top three spaces with the highest choice."
            ),
            expected_metric_values=choice_gt,
            expected_top_3=rank_top_3(choice_gt, mode="highest"),
            top_3_key="top_3_highest_choice",
            ranking_rule="Higher choice ranks higher. Ties are broken alphabetically.",
            metadata={"metric": "choice_unnormalized", "ranking": "highest", "tolerance": REL_TOLERANCE},
        ))
    return tasks

# Prompting

def semantic_rule_set(task):
    if task.task_type == "T4_mean_depth":
        return {
            "goal": "Evaluate global spatial integration/reachability.",
            "principle": "Mean depth is average shortest-path distance from a space to all other spaces. Lower is more integrated.",
            "use": ["calculate_mean_depth", "get_node_profile", "submit_answer"],
            "ranking_rule": task.ranking_rule,
        }
    if task.task_type == "T5_choice":
        return {
            "goal": "Evaluate through-movement importance and circulation bottlenecks.",
            "principle": "Choice is unnormalized betweenness/number of shortest paths passing through a space. Higher is more important.",
            "use": ["calculate_choice", "get_circulation_spaces", "get_node_profile", "submit_answer"],
            "ranking_rule": task.ranking_rule,
        }
    raise ValueError(f"Unknown task type: {task.task_type}")

SYSTEM_PROMPT = """
You are a graph-reasoning benchmark participant.
You must use the provided metric tools and semantic KG tools when useful.
You must submit the final answer by calling the submit_answer tool.
Do not answer in free text.
Use exact node names.
""".strip()

def build_prompt(task):
    adjacency = graphs[task.graph_name]
    triples = build_semantic_kg_triples(adjacency)
    output_format = {
        "metric_values": {"space_name": 0.0},
        task.top_3_key: ["space_1", "space_2", "space_3"],
    }
    return f"""
<benchmark_task>
  <role>You are evaluating a room connectivity graph with semantic KG support. Treat the graph as undirected.</role>
  <graph_name>{task.graph_name}</graph_name>
  <connectivity_graph>
{graph_to_json(adjacency)}
  </connectivity_graph>
  <semantic_kg_triples>
{kg_triples_to_json(triples)}
  </semantic_kg_triples>
  <semantic_rule>
{json.dumps(semantic_rule_set(task), indent=2)}
  </semantic_rule>
  <question>{task.question}</question>
  <instructions>
    <instruction>Use graph_name exactly as provided.</instruction>
    <instruction>Use exact node names returned by tools.</instruction>
    <instruction>Include every node returned by the metric tool in metric_values.</instruction>
    <instruction>Round metric values to 4 decimal places.</instruction>
    <instruction>{task.ranking_rule}</instruction>
    <instruction>Use submit_answer for the final answer.</instruction>
  </instructions>
  <required_output_format>
{json.dumps(output_format, indent=2)}
  </required_output_format>
</benchmark_task>
""".strip()

# Tool schemas and execution

def make_tools(task):
    return [
        {
            "name": "calculate_mean_depth",
            "description": "Calculate mean depth for every node in a named graph.",
            "input_schema": {
                "type": "object",
                "properties": {"graph_name": {"type": "string", "enum": list(graphs.keys())}},
                "required": ["graph_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "calculate_choice",
            "description": "Calculate unnormalized choice for every node in a named graph.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string", "enum": list(graphs.keys())},
                    "normalized": {"type": "boolean", "default": False},
                },
                "required": ["graph_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_node_profile",
            "description": "Return semantic KG profile and degree/neighbors for a node.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string", "enum": list(graphs.keys())},
                    "node": {"type": "string"},
                },
                "required": ["graph_name", "node"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_spaces_by_role",
            "description": "Return spaces with a semantic role from the KG.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string", "enum": list(graphs.keys())},
                    "role": {
                        "type": "string",
                        "enum": ["entry", "private_space", "shared_or_guest_facing_space", "circulation"],
                    },
                },
                "required": ["graph_name", "role"],
                "additionalProperties": False,
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final structured graph metric benchmark answer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "metric_values": {"type": "object", "additionalProperties": {"type": "number"}},
                    task.top_3_key: {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
                },
                "required": ["metric_values", task.top_3_key],
                "additionalProperties": False,
            },
        },
    ]

def execute_tool(tool_name, tool_input):
    graph_name = tool_input["graph_name"]
    adjacency = graphs[graph_name]
    triples = build_semantic_kg_triples(adjacency)

    if tool_name == "calculate_mean_depth":
        return Tools.calculate_mean_depth(adjacency)

    if tool_name == "calculate_choice":
        return calculate_choice(adjacency, normalized=bool(tool_input.get("normalized", False)))

    if tool_name == "get_node_profile":
        node = tool_input["node"]
        return {
            "node": node,
            "degree": len(adjacency.get(node, [])),
            "neighbors": sorted(adjacency.get(node, [])),
            "has_type": tails_for_head(triples, node, "has_type"),
            "has_privacy_category": tails_for_head(triples, node, "has_privacy_category"),
            "has_access_role": tails_for_head(triples, node, "has_access_role"),
            "has_role": tails_for_head(triples, node, "has_role"),
        }

    if tool_name == "get_spaces_by_role":
        role = tool_input["role"]
        if role == "entry":
            return get_entries_from_kg(triples)
        if role == "private_space":
            return get_private_spaces_from_kg(triples)
        if role == "shared_or_guest_facing_space":
            return get_shared_or_guest_facing_spaces_from_kg(triples)
        if role == "circulation":
            return get_circulation_spaces_from_kg(triples)

    raise ValueError(f"Unknown executable tool: {tool_name}")

# Claude API tool-use loop with forced submit_answer

def call(api_key, system, user, task):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user}]
        force_submit_next = False
        try:
            while True:
                tools = make_tools(task)
                if force_submit_next:
                    tools = [tool for tool in tools if tool.get("name") == "submit_answer"]

                payload: Dict[str, Any] = {
                    "model": MODEL_NAME,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "system": system,
                    "messages": messages,
                    "tools": tools,
                }
                if force_submit_next:
                    payload["tool_choice"] = {"type": "tool", "name": "submit_answer"}

                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                if response.status_code == 401:
                    raise RuntimeError(f"401 Unauthorized: {response.text}")
                if response.status_code == 429:
                    raise requests.HTTPError("429 Rate limit", response=response)
                response.raise_for_status()
                data = response.json()

                stop_reason = data.get("stop_reason")
                content_blocks = data.get("content", [])
                tool_uses = [block for block in content_blocks if block.get("type") == "tool_use"]
                messages.append({"role": "assistant", "content": content_blocks})

                if tool_uses:
                    tool_results = []
                    for tool_use in tool_uses:
                        tool_name = tool_use["name"]
                        tool_input = tool_use.get("input", {})
                        tool_use_id = tool_use["id"]

                        if tool_name == "submit_answer":
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
                    messages.append({
                        "role": "user",
                        "content": "Continue until complete. Use submit_answer for the final structured answer.",
                    })
                    continue

                if stop_reason == "end_turn":
                    force_submit_next = True
                    messages.append({
                        "role": "user",
                        "content": (
                            "You must now call submit_answer with the final structured answer. "
                            "Do not provide free text."
                        ),
                    })
                    continue

                messages.append({
                    "role": "user",
                    "content": "Continue until complete. Use submit_answer for the final structured answer.",
                })

        except requests.HTTPError as exc:
            last_error = exc
            status = getattr(exc.response, "status_code", None)
            if status in {400, 401, 403, 404}:
                raise
            time.sleep(WAIT_SECONDS * (2 ** (attempt - 1)))
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(WAIT_SECONDS * (2 ** (attempt - 1)))
        except RuntimeError as exc:
            last_error = exc
            if "401 Unauthorized" in str(exc):
                raise
            time.sleep(WAIT_SECONDS * (2 ** (attempt - 1)))

    raise RuntimeError(f"Claude API call failed after {MAX_RETRIES} retries: {last_error}")

# Parsing and scoring

def extract_json(raw_text):
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None

def normalize_metric_values(value):
    if not isinstance(value, dict):
        return None
    normalized: Dict[str, float] = {}
    for key, val in value.items():
        try:
            normalized[str(key).strip()] = float(val)
        except (TypeError, ValueError):
            return None
    return dict(sorted(normalized.items()))

def normalize_top3(value):
    if not isinstance(value, list) or len(value) != 3:
        return None
    return [str(x).strip() for x in value]

def value_within_tolerance(predicted, expected):
    if abs(expected) <= ABS_TOLERANCE_FOR_ZERO:
        return abs(predicted - expected) <= ABS_TOLERANCE_FOR_ZERO
    return abs(predicted - expected) / abs(expected) <= REL_TOLERANCE

def metric_node_accuracy(predicted, expected):
    if predicted is None:
        return 0.0, 0, len(expected), [], {node: None for node in expected}
    correct_nodes: List[str] = []
    rel_errors: Dict[str, Optional[float]] = {}
    for node, expected_value in expected.items():
        if node not in predicted:
            rel_errors[node] = None
            continue
        predicted_value = predicted[node]
        if abs(expected_value) <= ABS_TOLERANCE_FOR_ZERO:
            rel_errors[node] = 0.0 if abs(predicted_value - expected_value) <= ABS_TOLERANCE_FOR_ZERO else None
        else:
            rel_errors[node] = abs(predicted_value - expected_value) / abs(expected_value)
        if value_within_tolerance(predicted_value, expected_value):
            correct_nodes.append(node)
    total = len(expected)
    num_correct = len(correct_nodes)
    return (num_correct / total if total else 0.0), num_correct, total, correct_nodes, rel_errors

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

def score_answer(task, parsed):
    if parsed is None:
        return {
            "predicted_metric_values": None,
            "predicted_top_3": None,
            "metric_accuracy": 0.0,
            "metric_num_correct": 0,
            "metric_total_nodes": len(task.expected_metric_values),
            "metric_correct_nodes": [],
            "ranking_accuracy": 0.0,
            "ranking_num_correct": 0,
            "ranking_total_positions": len(task.expected_top_3),
            "ranking_correct_positions": [],
            "overall_correct": 0,
            "per_node_relative_errors": {node: None for node in task.expected_metric_values},
        }

    predicted_metric_values = normalize_metric_values(parsed.get("metric_values"))
    predicted_top_3 = normalize_top3(parsed.get(task.top_3_key))
    metric_acc, num_correct, total_nodes, correct_nodes, rel_errors = metric_node_accuracy(predicted_metric_values, task.expected_metric_values)
    ranking_acc, ranking_num_correct, ranking_total_positions, ranking_correct_positions = top3_accuracy(predicted_top_3, task.expected_top_3)
    overall_correct = int(metric_acc == 1.0 and predicted_top_3 == task.expected_top_3)

    return {
        "predicted_metric_values": predicted_metric_values,
        "predicted_top_3": predicted_top_3,
        "metric_accuracy": metric_acc,
        "metric_num_correct": num_correct,
        "metric_total_nodes": total_nodes,
        "metric_correct_nodes": correct_nodes,
        "ranking_accuracy": ranking_acc,
        "ranking_num_correct": ranking_num_correct,
        "ranking_total_positions": ranking_total_positions,
        "ranking_correct_positions": ranking_correct_positions,
        "overall_correct": overall_correct,
        "per_node_relative_errors": rel_errors,
    }

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

# Logging and summaries
def make_ground_truth_table(tasks):
    return pd.DataFrame([
        {
            "condition": "kg",
            "task_family": "global",
            "task_id": task.task_id,
            "task_number": task.task_number,
            "graph_name": task.graph_name,
            "task_type": task.task_type,
            "question": task.question,
            "expected_metric_values": json.dumps(task.expected_metric_values, ensure_ascii=False),
            "expected_top_3": json.dumps(task.expected_top_3, ensure_ascii=False),
            "top_3_key": task.top_3_key,
            "ranking_rule": task.ranking_rule,
            "metadata": json.dumps(task.metadata, ensure_ascii=False),
        }
    for task in tasks])

def summarize_results(results_df):
    rows = []
    group_cols = ["condition", "task_family", "graph_name", "task_id", "task_number", "task_type", "question", "expected_metric_values", "expected_top_3"]
    for keys, group in results_df.groupby(group_cols, dropna=False):
        condition, task_family, graph_name, task_id, task_number, task_type, question, expected_metric_values, expected_top_3 = keys
        overall_correct_values = group["overall_correct"].astype(int).tolist()
        predicted_top3_values = [json.loads(x) if pd.notna(x) else None for x in group["predicted_top_3"]]
        predicted_metric_values = [json.loads(x) if pd.notna(x) else None for x in group["predicted_metric_values"]]
        rows.append({
            "condition": condition,
            "task_family": task_family,
            "graph_name": graph_name,
            "task_id": task_id,
            "task_number": task_number,
            "task_type": task_type,
            "question": question,
            "expected_metric_values": expected_metric_values,
            "expected_top_3": expected_top_3,
            "n_runs": len(group),
            "mean_metric_accuracy": group["metric_accuracy"].astype(float).mean(),
            "mean_ranking_accuracy": group["ranking_accuracy"].astype(float).mean(),
            "overall_exact_pass_rate": sum(overall_correct_values) / len(overall_correct_values),
            "pass_at_5_overall": int(any(v == 1 for v in overall_correct_values)),
            "pass_5_all_correct_overall": int(all(v == 1 for v in overall_correct_values)),
            "consistency_overall_correctness": binary_consistency(overall_correct_values),
            "consistency_top3_answer": consistency_score(predicted_top3_values),
            "consistency_metric_values": consistency_score(predicted_metric_values),
        })
    return pd.DataFrame(rows).sort_values(["graph_name", "task_id"])

def summarize_by_task_type(question_summary_df):
    
    rows = []
    for task_type, group in question_summary_df.groupby("task_type", dropna=False):
        rows.append({
            "condition": "kg",
            "task_family": "global",
            "task_type": task_type,
            "n_questions": int(len(group)),
            "mean_metric_accuracy": group["mean_metric_accuracy"].astype(float).mean(),
            "mean_ranking_accuracy": group["mean_ranking_accuracy"].astype(float).mean(),
            "overall_exact_pass_rate": group["overall_correct"].astype(float).mean() if "overall_correct" in group.columns else None,
            "pass_at_5": group["pass_at_5_overall"].astype(float).mean(),
            "pass_5_all_correct": group["pass_5_all_correct_overall"].astype(float).mean(),
            "consistency_overall_correctness": group["consistency_overall_correctness"].astype(float).mean(),
            "consistency_top3_answer": group["consistency_top3_answer"].astype(float).mean(),
            "consistency_metric_values": group["consistency_metric_values"].astype(float).mean(),
        })
    return pd.DataFrame(rows).sort_values(["task_type"])

# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api_key = get_api_key()
    tasks = make_tasks()

    gt_df = make_ground_truth_table(tasks)
    gt_path = export_csv(gt_df, OUTPUT_DIR, MODULE_NAME, "ground_truth")
    print(f"Ground truth saved to: {gt_path}")
    print(gt_df[["task_id", "question", "expected_top_3"]].to_string(index=False))

    result_rows: List[Dict[str, Any]] = []
    total_calls = len(tasks) * N_RUNS
    call_idx = 0

    for task in tasks:
        prompt = build_prompt(task)
        for run_idx in range(1, N_RUNS + 1):
            call_idx += 1
            print(f"\n[{call_idx}/{total_calls}] {task.task_id} run {run_idx}/{N_RUNS}")
            raw_answer = call(api_key, SYSTEM_PROMPT, prompt, task)
            parsed = extract_json(raw_answer)
            scored = score_answer(task, parsed)
            row = {
                "condition": "kg",
                "task_family": "global",
                "task_id": task.task_id,
                "task_number": task.task_number,
                "task_type": task.task_type,
                "graph_name": task.graph_name,
                "run_idx": run_idx,
                "question": task.question,
                "expected_metric_values": json.dumps(task.expected_metric_values, ensure_ascii=False),
                "expected_top_3": json.dumps(task.expected_top_3, ensure_ascii=False),
                "raw_answer": raw_answer,
                "parsed_answer": json.dumps(parsed, ensure_ascii=False) if parsed is not None else None,
                "predicted_metric_values": json.dumps(scored["predicted_metric_values"], ensure_ascii=False),
                "predicted_top_3": json.dumps(scored["predicted_top_3"], ensure_ascii=False),
                "metric_accuracy": scored["metric_accuracy"],
                "metric_num_correct": scored["metric_num_correct"],
                "metric_total_nodes": scored["metric_total_nodes"],
                "metric_correct_nodes": json.dumps(scored["metric_correct_nodes"], ensure_ascii=False),
                "ranking_accuracy": scored["ranking_accuracy"],
                "ranking_num_correct": scored["ranking_num_correct"],
                "ranking_total_positions": scored["ranking_total_positions"],
                "ranking_correct_positions": json.dumps(scored["ranking_correct_positions"], ensure_ascii=False),
                "overall_correct": scored["overall_correct"],
                "per_node_relative_errors": json.dumps(scored["per_node_relative_errors"], ensure_ascii=False),
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "rel_tolerance": REL_TOLERANCE,
            }
            result_rows.append(row)
            export_csv(pd.DataFrame(result_rows), OUTPUT_DIR, MODULE_NAME, "results_partial")
            print(
                f"Metric accuracy: {scored['metric_num_correct']}/{scored['metric_total_nodes']} ({scored['metric_accuracy']:.2%}) | "
                f"Ranking accuracy: {scored['ranking_num_correct']}/{scored['ranking_total_positions']} ({scored['ranking_accuracy']:.2%}) | "
                f"Overall: {scored['overall_correct']}"
            )
            print(f"Predicted top 3: {scored['predicted_top_3']}")
            time.sleep(WAIT_SECONDS)

    results_df = pd.DataFrame(result_rows)
    results_path = export_csv(results_df, OUTPUT_DIR, MODULE_NAME, "results")
    summary_df = summarize_results(results_df)
    summary_path = export_csv(summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_question")
    type_summary_df = summarize_by_task_type(summary_df)
    type_summary_path = export_csv(type_summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_type")
    raw_log_path = export_csv(
        results_df[["task_id", "task_type", "graph_name", "run_idx", "raw_answer", "parsed_answer", "expected_top_3", "predicted_top_3", "metric_accuracy", "ranking_accuracy", "overall_correct"]],
        OUTPUT_DIR,
        MODULE_NAME,
        "raw_answers",
    )

    print("\nSaved outputs:")
    for path in [gt_path, results_path, summary_path, type_summary_path, raw_log_path]:
        print(f"- {path}")
    print("\nSummary by question:")
    print(summary_df.to_string(index=False))
    print("\nSummary by task type:")
    print(type_summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
