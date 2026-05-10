from __future__ import annotations

"""
Counterfactual graph-reasoning benchmark for Claude with semantic KG + metric tools.

What this script does:
1. Imports graph_1 ... graph_5 from connectivity_graph_data.py.
2. Applies a counterfactual edit: add new_bedroom connected to the configured attachment node.
3. Computes deterministic ground truth for two counterfactual tasks:
      Task 6: changed top 3 lowest mean depth spaces
      Task 7: changed top 3 highest choice spaces
4. Calls Claude with real tool_use:
      - calculate_mean_depth
      - calculate_choice
      - semantic KG tools: get_entries, get_private_spaces,
        get_shared_or_guest_facing_spaces, get_circulation_spaces
      - submit_answer for final structured output
5. Loops through tool_use rounds until stop_reason == "end_turn".
6. Keeps parsing, scoring, logging, and summary reporting structure from the prior counterfactual script.

"""

import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

import connectivity_graph_data
import Tools

from kg_build import (
    Triple,
    build_semantic_kg_triples,
    get_circulation_spaces_from_kg,
    get_entries_from_kg,
    get_private_spaces_from_kg,
    get_shared_or_guest_facing_spaces_from_kg,
)
from export_utils import export_csv, export_csvs, module_name, output_dir_for, print_saved_outputs

# Config

MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
N_RUNS = int(os.getenv("N_RUNS", "5"))
WAIT_SECONDS = float(os.getenv("WAIT_SECONDS", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "900"))
MODULE_NAME = module_name(__file__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", SCRIPT_DIR)
NEW_NODE = "new_bedroom"

graphs: Dict[str, Dict[str, List[str]]] = {
    "graph_1": connectivity_graph_data.graph_1,
    "graph_2": connectivity_graph_data.graph_2,
    "graph_3": connectivity_graph_data.graph_3,
    "graph_4": connectivity_graph_data.graph_4,
    "graph_5": connectivity_graph_data.graph_5,
}

ATTACH_TO = {
    "graph_1": "hall",
    "graph_2": "hall",
    "graph_3": "hall",
    "graph_4": "hall",
    "graph_5": "hall_1",
}

@dataclass
class Task:
    task_id: str
    task_type: str
    graph_name: str
    question: str
    expected_answer: Dict[str, Any]

# Api key
def get_api_key():
    if os.getenv("SKIP_API_KEY_PROMPT", "0") == "1":
        key = os.getenv("ANTHROPIC_API_KEY", "")
    else:
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key.strip():
            print("Enter API key:")
            key = input().strip().strip('"').strip("'")

    key = key.strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("Missing API key")
    os.environ["ANTHROPIC_API_KEY"] = key
    print(f"API key loaded. Length: {len(key)} characters")
    return key

# Graph utils
def add_new_bedroom(adj, attach):
    g = deepcopy(adj)
    if attach not in g:
        raise ValueError(f"Attachment node not found: {attach}")
    if NEW_NODE not in g[attach]:
        g[attach].append(NEW_NODE)
    g[NEW_NODE] = [attach]
    return g

def counterfactual_graph(graph_name):
    return add_new_bedroom(graphs[graph_name], ATTACH_TO[graph_name])

# Compact semantic rule set

SEMANTIC_RULE_SET = {
    "counterfactual_mean_depth": {
        "goal": "Assess how the edited layout changes global spatial integration.",
        "principle": "After applying the counterfactual graph edit, lower mean depth means a space is more integrated/reachable.",
        "use": ["calculate_mean_depth", "submit_answer"],
    },
    "counterfactual_choice": {
        "goal": "Assess how the edited layout changes movement concentration or through-movement importance.",
        "principle": "After applying the counterfactual graph edit, higher choice means a space lies on more shortest paths.",
        "use": ["calculate_choice", "get_circulation_spaces", "submit_answer"],
    },
}

TASK_TO_RULE_KEY = {
    "Task_6_counterfactual_mean_depth": "counterfactual_mean_depth",
    "Task_7_counterfactual_choice": "counterfactual_choice",
}

# Metric calls
def mean_depth(adj):
    return Tools.calculate_mean_depth(adj)

def choice(adj):
    return Tools.calculate_choice(adj, normalized=False)

def top3(values, reverse = False):
    return [
        k for k, _ in sorted(
            values.items(),
            key=lambda x: (-float(x[1]), str(x[0])) if reverse else (float(x[1]), str(x[0])),
        )[:3]
    ]

# Ground truth

def compute_ground_truth():
    gt = {}

    for name, g in graphs.items():
        g2 = add_new_bedroom(g, ATTACH_TO[name])

        md = mean_depth(g2)
        ch = choice(g2)

        gt[name] = {
            "Task_6_counterfactual_mean_depth": {
                "changed_top_3_lowest_mean_depth": top3(md)
            },
            "Task_7_counterfactual_choice": {
                "changed_top_3_highest_choice": top3(ch, reverse=True)
            }
        }

    return gt

# Task generation

def make_tasks(gt):
    tasks = []

    for gname in graphs.keys():

        tasks.append(Task(
            task_id=f"{gname}_T6",
            task_type="Task_6_counterfactual_mean_depth",
            graph_name=gname,
            question=f"After adding a new bedroom to {ATTACH_TO[gname]}, return top 3 lowest mean depth spaces.",
            expected_answer=gt[gname]["Task_6_counterfactual_mean_depth"]
        ))

        tasks.append(Task(
            task_id=f"{gname}_T7",
            task_type="Task_7_counterfactual_choice",
            graph_name=gname,
            question=f"After adding a new bedroom to {ATTACH_TO[gname]}, return top 3 highest choice spaces.",
            expected_answer=gt[gname]["Task_7_counterfactual_choice"]
        ))

    return tasks

# Prompts
SYSTEM_PROMPT = """
You are a graph reasoning benchmark participant.
You must use the provided tools after applying the requested counterfactual change.
Use the semantic KG tools when semantic categories or roles are useful.
You must submit the final answer by calling the submit_answer tool.
Do not answer in free text.
""".strip()

def final_answer_key(task):
    if "mean_depth" in task.task_type:
        return "changed_top_3_lowest_mean_depth"
    return "changed_top_3_highest_choice"

def build_prompt(task):
    attach = ATTACH_TO[task.graph_name]
    output_key = final_answer_key(task)
    rule_key = TASK_TO_RULE_KEY[task.task_type]
    semantic_rule = SEMANTIC_RULE_SET[rule_key]

    return f"""
<benchmark_task>
  <graph_name>{task.graph_name}</graph_name>

  <counterfactual_change>
    Add {NEW_NODE} connected to {attach}.
  </counterfactual_change>

  <question>
    {task.question}
  </question>

  <semantic_rule>
{json.dumps(semantic_rule, indent=2)}
  </semantic_rule>

  <instructions>
    <instruction>The tools operate on the counterfactual graph after adding {NEW_NODE}.</instruction>
    <instruction>Use exact node names returned by the tools.</instruction>
    <instruction>Break metric ties alphabetically by node name.</instruction>
    <instruction>Submit the final answer using submit_answer.</instruction>
  </instructions>

  <required_output_format>
{{
  "{output_key}": ["space_1", "space_2", "space_3"]
}}
  </required_output_format>
</benchmark_task>
""".strip()

# Api tool schemas and execution

def make_tools(task):
    answer_key = final_answer_key(task)

    return [
        {
            "name": "calculate_mean_depth",
            "description": "Calculate mean depth for every node in the counterfactual graph.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {
                        "type": "string",
                        "enum": list(graphs.keys()),
                    }
                },
                "required": ["graph_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "calculate_choice",
            "description": "Calculate unnormalized choice for every node in the counterfactual graph.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {
                        "type": "string",
                        "enum": list(graphs.keys()),
                    },
                    "normalized": {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "required": ["graph_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_entries",
            "description": "Return valid entry spaces in the counterfactual graph using the semantic KG.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {
                        "type": "string",
                        "enum": list(graphs.keys()),
                    }
                },
                "required": ["graph_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_private_spaces",
            "description": "Return private spaces in the counterfactual graph using the semantic KG.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {
                        "type": "string",
                        "enum": list(graphs.keys()),
                    }
                },
                "required": ["graph_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_shared_or_guest_facing_spaces",
            "description": "Return shared or guest-facing spaces in the counterfactual graph using the semantic KG.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {
                        "type": "string",
                        "enum": list(graphs.keys()),
                    }
                },
                "required": ["graph_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_circulation_spaces",
            "description": "Return circulation spaces in the counterfactual graph using the semantic KG.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {
                        "type": "string",
                        "enum": list(graphs.keys()),
                    }
                },
                "required": ["graph_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final structured counterfactual top-3 answer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    answer_key: {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 3,
                    }
                },
                "required": [answer_key],
                "additionalProperties": False,
            },
        },
    ]

def execute_tool(tool_name, tool_input):
    graph_name = tool_input["graph_name"]
    cf_graph = counterfactual_graph(graph_name)

    if tool_name == "calculate_mean_depth":
        return Tools.calculate_mean_depth(cf_graph)

    if tool_name == "calculate_choice":
        return Tools.calculate_choice(
            cf_graph,
            normalized=bool(tool_input.get("normalized", False)),
        )

    triples = build_semantic_kg_triples(cf_graph)

    if tool_name == "get_entries":
        return get_entries_from_kg(triples)

    if tool_name == "get_private_spaces":
        return get_private_spaces_from_kg(triples)

    if tool_name == "get_shared_or_guest_facing_spaces":
        return get_shared_or_guest_facing_spaces_from_kg(triples)

    if tool_name == "get_circulation_spaces":
        return get_circulation_spaces_from_kg(triples)

    raise ValueError(f"Unknown executable tool: {tool_name}")

# Api call

def call(api_key, system, user, task):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        messages = [{"role": "user", "content": user}]
        submitted_answer: Optional[Dict[str, Any]] = None
        force_submit_next = False
        try:
            while True:
                tools = make_tools(task)
                if force_submit_next:
                    tools = [tool for tool in tools if tool.get("name") == "submit_answer"]

                payload = {
                    "model": MODEL_NAME,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "system": system,
                    "messages": messages,
                    "tools": tools,
                }
                if force_submit_next:
                    payload["tool_choice"] = {"type": "tool", "name": "submit_answer"}

                response = requests.post(
                    API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 401:
                    raise RuntimeError(f"401 Unauthorized: {response.text}")

                if response.status_code == 429:
                    raise requests.HTTPError("429 Rate limit", response=response)

                response.raise_for_status()
                data = response.json()

                stop_reason = data.get("stop_reason")
                content_blocks = data.get("content", [])

                tool_uses = [
                    block for block in content_blocks
                    if block.get("type") == "tool_use"
                ]

                messages.append({
                    "role": "assistant",
                    "content": content_blocks,
                })

                if tool_uses:
                    tool_results = []

                    for tool_use in tool_uses:
                        tool_name = tool_use["name"]
                        tool_input = tool_use.get("input", {})
                        tool_use_id = tool_use["id"]

                        if tool_name == "submit_answer":
                            submitted_answer = tool_input
                            return json.dumps(submitted_answer, ensure_ascii=False)

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

                    messages.append({
                        "role": "user",
                        "content": tool_results,
                    })

                    continue

                if stop_reason == "max_tokens":
                    messages.append({
                        "role": "user",
                        "content": "Continue until complete. Use submit_answer for the final structured answer.",
                    })
                    continue

                if stop_reason == "end_turn":
                    if submitted_answer is not None:
                        return json.dumps(submitted_answer, ensure_ascii=False)

                    force_submit_next = True
                    messages.append({
                        "role": "user",
                        "content": (
                            "You must now call the submit_answer tool. "
                            "Do not provide free text. Return only the structured submit_answer tool input."
                        ),
                    })
                    continue

                messages.append({
                    "role": "user",
                    "content": "Continue until complete. Use submit_answer for the final structured answer.",
                })

        except requests.exceptions.HTTPError as exc:
            last_error = exc
            status = getattr(exc.response, "status_code", None)
            text = getattr(exc.response, "text", "")
            print(f"HTTP error on attempt {attempt}/{MAX_RETRIES}: {status} {text[:300]}")

            if status in {400, 401, 403, 404}:
                raise

            time.sleep(WAIT_SECONDS * (2 ** (attempt - 1)))

        except requests.exceptions.RequestException as exc:
            last_error = exc
            print(f"Request error on attempt {attempt}/{MAX_RETRIES}: {exc}")
            time.sleep(WAIT_SECONDS * (2 ** (attempt - 1)))

        except RuntimeError as exc:
            last_error = exc
            if "401 Unauthorized" in str(exc):
                raise
            print(f"Runtime error on attempt {attempt}/{MAX_RETRIES}: {exc}")
            time.sleep(WAIT_SECONDS * (2 ** (attempt - 1)))

    raise RuntimeError(f"Claude API call failed after {MAX_RETRIES} retries: {last_error}")

# Parser

def extract(tag, text):
    try:
        return re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL).group(1).strip()
    except:
        return None

def parse(text, task):
    if not text:
        return None

    key = final_answer_key(task)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and key in parsed:
            value = parsed[key]
            if isinstance(value, list):
                return [str(x).strip() for x in value]
    except json.JSONDecodeError:
        pass

    if "mean_depth" in task.task_type:
        val = extract("changed_top_3_lowest_mean_depth", text)
    else:
        val = extract("changed_top_3_highest_choice", text)

    if not val:
        return None

    return [x.strip() for x in val.split(",")]

# Scoring

def expected_top3(task):
    return list(task.expected_answer.values())[0]

def top3_accuracy(predicted, expected):
    """
    Position-wise top-3 accuracy.
    This avoids reporting 0 for an answer that is partly correct but has one wrong/misordered item.
    """
    if predicted is None:
        return 0.0, 0, len(expected), []

    correct_positions = [
        f"pos_{idx + 1}:{gt}"
        for idx, (pred, gt) in enumerate(zip(predicted, expected))
        if pred == gt
    ]
    num_correct = len(correct_positions)
    total = len(expected)
    percent = num_correct / total if total else 0.0
    return percent, num_correct, total, correct_positions

def score_answer(task, parsed):
    expected = expected_top3(task)
    ranking_acc, num_correct, total_positions, correct_positions = top3_accuracy(parsed, expected)
    overall_correct = int(parsed == expected)
    return {
        "predicted_top_3": parsed,
        "ranking_accuracy": ranking_acc,
        "ranking_num_correct": num_correct,
        "ranking_total_positions": total_positions,
        "ranking_correct_positions": correct_positions,
        "overall_correct": overall_correct,
    }

def pass_at_k(scores):
    return int(any(scores))

def pass_all(scores):
    return int(all(scores))

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
    rows = []
    for task in tasks:
        rows.append({
            "task_id": task.task_id,
            "graph_name": task.graph_name,
            "task_type": task.task_type,
            "question": task.question,
            "expected_top_3": json.dumps(expected_top3(task), ensure_ascii=False),
            "expected_answer": json.dumps(task.expected_answer, ensure_ascii=False),
        })
    return pd.DataFrame(rows)

def summarize_results(results_df):
    rows = []
    group_cols = ["graph_name", "task_id", "task_type", "question", "expected_top_3"]
    for keys, group in results_df.groupby(group_cols, dropna=False):
        graph_name, task_id, task_type, question, expected_top_3 = keys
        ranking_accuracies = group["ranking_accuracy"].astype(float).tolist()
        overall_correct_values = group["overall_correct"].astype(int).tolist()
        predicted_top3_values = []
        for _, row in group.iterrows():
            try:
                predicted_top3_values.append(json.loads(row["predicted_top_3"]) if pd.notna(row["predicted_top_3"]) else None)
            except Exception:
                predicted_top3_values.append(None)
        rows.append({
            "graph_name": graph_name,
            "task_id": task_id,
            "task_type": task_type,
            "question": question,
            "expected_top_3": expected_top_3,
            "n_runs": len(group),
            "mean_ranking_accuracy": sum(ranking_accuracies) / len(ranking_accuracies),
            "overall_exact_pass_rate": sum(overall_correct_values) / len(overall_correct_values),
            "pass_at_5_overall": pass_at_k(overall_correct_values),
            "pass_5_all_correct_overall": pass_all(overall_correct_values),
            "consistency_overall_correctness": binary_consistency(overall_correct_values),
            "consistency_top3_answer": consistency_score(predicted_top3_values),
        })
    return pd.DataFrame(rows).sort_values(["graph_name", "task_id"])

def summarize_by_task_type(question_summary_df):
    
    rows = []
    for task_type, group in question_summary_df.groupby("task_type", dropna=False):
        rows.append({
            "task_type": task_type,
            "n_questions": int(len(group)),
            "mean_ranking_accuracy": group["mean_ranking_accuracy"].astype(float).mean(),
            "overall_exact_pass_rate": group["overall_exact_pass_rate"].astype(float).mean(),
            "pass_at_5": group["pass_at_5_overall"].astype(float).mean(),
            "pass_5_all_correct": group["pass_5_all_correct_overall"].astype(float).mean(),
            "consistency_overall_correctness": group["consistency_overall_correctness"].astype(float).mean(),
            "consistency_top3_answer": group["consistency_top3_answer"].astype(float).mean(),
        })
    return pd.DataFrame(rows).sort_values(["task_type"])

# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    api = get_api_key()
    gt = compute_ground_truth()
    tasks = make_tasks(gt)

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

            raw_answer = call(api, SYSTEM_PROMPT, prompt, task)
            parsed = parse(raw_answer, task)
            scored = score_answer(task, parsed)

            expected = expected_top3(task)
            row = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "graph_name": task.graph_name,
                "run_idx": run_idx,
                "question": task.question,
                "expected_top_3": json.dumps(expected, ensure_ascii=False),
                "raw_answer": raw_answer,
                "parsed_answer": json.dumps(parsed, ensure_ascii=False) if parsed is not None else None,
                "predicted_top_3": json.dumps(scored["predicted_top_3"], ensure_ascii=False),
                "ranking_accuracy": scored["ranking_accuracy"],
                "ranking_num_correct": scored["ranking_num_correct"],
                "ranking_total_positions": scored["ranking_total_positions"],
                "ranking_correct_positions": json.dumps(scored["ranking_correct_positions"], ensure_ascii=False),
                "overall_correct": scored["overall_correct"],
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
            }
            result_rows.append(row)

            partial_df = pd.DataFrame(result_rows)
            partial_path = export_csv(partial_df, OUTPUT_DIR, MODULE_NAME, "results_partial")

            print("Raw answer:")
            print(raw_answer)
            print(
                "Ranking accuracy: "
                f"{scored['ranking_num_correct']}/{scored['ranking_total_positions']} "
                f"({scored['ranking_accuracy']:.2%}) | "
                f"Overall: {scored['overall_correct']}"
            )
            print(f"Expected top 3: {expected}")
            print(f"Predicted top 3: {scored['predicted_top_3']}")

            time.sleep(WAIT_SECONDS)

    results_df = pd.DataFrame(result_rows)
    summary_df = summarize_results(results_df)
    type_summary_df = summarize_by_task_type(summary_df)
    raw_log_df = results_df[[
        "task_id", "task_type", "graph_name", "run_idx", "raw_answer", "parsed_answer",
        "expected_top_3", "predicted_top_3", "ranking_accuracy", "overall_correct"
    ]]

    final_paths = export_csvs({
        "results": results_df,
        "summary_by_question": summary_df,
        "summary_by_type": type_summary_df,
        "raw_answers": raw_log_df,
    }, OUTPUT_DIR, MODULE_NAME)

    print_saved_outputs([gt_path, *final_paths.values()])

    print("\nSummary by question:")
    print(summary_df.to_string(index=False))

    print("\nSummary by task type:")
    print(type_summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
