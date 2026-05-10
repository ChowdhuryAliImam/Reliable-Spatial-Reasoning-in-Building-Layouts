from __future__ import annotations
from tool_call_helper import call_with_tool_use, get_api_key as helper_get_api_key
from export_utils import ensure_output_dir, export_csv

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

MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
N_RUNS = int(os.getenv("N_RUNS", "5"))
WAIT_SECONDS = float(os.getenv("WAIT_SECONDS", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3000"))
API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MODULE_NAME = "counterfactual_tool"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", SCRIPT_DIR)
NEW_NODE = "new_bedroom"

graphs = {
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

# Task class

@dataclass
class Task:
    task_id: str
    task_type: str
    graph_name: str
    question: str
    expected_answer: Dict[str, Any]

# Api key
def get_api_key():
    return helper_get_api_key("Enter API key:")

# Graph utils
def add_new_bedroom(adj, attach):
    g = deepcopy(adj)
    if attach not in g:
        raise ValueError(f"Attach node not found in graph: {attach}")
    if NEW_NODE not in g[attach]:
        g[attach].append(NEW_NODE)
    g[NEW_NODE] = [attach]
    return g

# Metric calls
def mean_depth(adj):
    return Tools.calculate_mean_depth(adj)

def choice(adj):
    return Tools.calculate_choice(adj, normalized=False)

def top3(values, reverse=False):
    return [
        k
        for k, _ in sorted(
            values.items(),
            key=lambda x: (-x[1], x[0]) if reverse else (x[1], x[0]),
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
            },
        }

    return gt

# Task generation
def make_tasks(gt):
    tasks = []

    for gname in graphs.keys():
        attach = ATTACH_TO[gname]

        tasks.append(Task(
            task_id=f"{gname}_T6",
            task_type="Task_6_counterfactual_mean_depth",
            graph_name=gname,
            question=f"After adding a new bedroom to {attach}, return top 3 lowest mean depth spaces.",
            expected_answer=gt[gname]["Task_6_counterfactual_mean_depth"],
        ))

        tasks.append(Task(
            task_id=f"{gname}_T7",
            task_type="Task_7_counterfactual_choice",
            graph_name=gname,
            question=f"After adding a new bedroom to {attach}, return top 3 highest choice spaces.",
            expected_answer=gt[gname]["Task_7_counterfactual_choice"],
        ))

    return tasks

# Prompts

SYSTEM_PROMPT = """
You are a graph reasoning benchmark participant.
You must use the provided metric tool after applying the requested counterfactual change.
Then you must submit the final answer using submit_answer.
Do not include explanations or markdown.
""".strip()

def final_answer_key(task):
    if "mean_depth" in task.task_type:
        return "changed_top_3_lowest_mean_depth"
    return "changed_top_3_highest_choice"

def build_prompt(task):
    attach = ATTACH_TO[task.graph_name]

    if "mean_depth" in task.task_type:
        output_key = "changed_top_3_lowest_mean_depth"
        metric_instruction = "Use calculate_mean_depth on the graph after adding new_bedroom."
        ranking_rule = "Return the top 3 spaces with the lowest mean depth. Ties are broken alphabetically."
    else:
        output_key = "changed_top_3_highest_choice"
        metric_instruction = "Use calculate_choice with normalized=false on the graph after adding new_bedroom."
        ranking_rule = "Return the top 3 spaces with the highest choice. Ties are broken alphabetically."

    return f"""
<benchmark_task>
  <graph_name>{task.graph_name}</graph_name>

  <counterfactual_change>
    Add {NEW_NODE} connected to {attach}.
  </counterfactual_change>

  <question>
    {task.question}
  </question>

  <instructions>
    <instruction>{metric_instruction}</instruction>
    <instruction>{ranking_rule}</instruction>
    <instruction>Do not invent spaces.</instruction>
    <instruction>Use exact node names returned by the tool.</instruction>
    <instruction>Submit the final answer using submit_answer.</instruction>
  </instructions>

  <required_output_format>
{{
  "{output_key}": ["space_1", "space_2", "space_3"]
}}
  </required_output_format>
</benchmark_task>
""".strip()

# Tool schemas and execution

def make_tools(task):
    answer_key = final_answer_key(task)

    return [
        {
            "name": "calculate_mean_depth",
            "description": "Calculate mean depth for every node in the counterfactual graph after adding new_bedroom.",
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
            "description": "Calculate choice for every node in the counterfactual graph after adding new_bedroom.",
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
    counterfactual_graph = add_new_bedroom(graphs[graph_name], ATTACH_TO[graph_name])

    if tool_name == "calculate_mean_depth":
        return Tools.calculate_mean_depth(counterfactual_graph)

    if tool_name == "calculate_choice":
        return Tools.calculate_choice(
            counterfactual_graph,
            normalized=bool(tool_input.get("normalized", False)),
        )

    raise ValueError(f"Unknown executable tool: {tool_name}")

# Api call
def call(api_key, system, user, task):
    return call_with_tool_use(
        api_key=api_key,
        user_prompt=user,
        system_prompt=system,
        tools=make_tools(task),
        execute_tool=execute_tool,
        model_name=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        api_url=API_URL,
        anthropic_version=ANTHROPIC_VERSION,
        max_retries=MAX_RETRIES,
        wait_seconds=WAIT_SECONDS,
        submit_tool_name="submit_answer",
        return_submitted_answer=True,
    )

# Parser
def extract(tag, text):
    try:
        return re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL).group(1).strip()
    except Exception:
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

    # Backward-compatible XML fallback
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
    """Aggregate only from per-question/task summary rows."""
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
    ensure_output_dir(OUTPUT_DIR)

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
    results_path = export_csv(results_df, OUTPUT_DIR, MODULE_NAME, "results")

    summary_df = summarize_results(results_df)
    summary_path = export_csv(summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_question")

    type_summary_df = summarize_by_task_type(summary_df)
    type_summary_path = export_csv(type_summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_type")

    raw_log_path = export_csv(
        results_df[[
            "task_id", "task_type", "graph_name", "run_idx", "raw_answer", "parsed_answer",
            "expected_top_3", "predicted_top_3", "ranking_accuracy", "overall_correct"
        ]],
        OUTPUT_DIR,
        MODULE_NAME,
        "raw_answers",
    )

    print("\nSaved outputs:")
    print(f"- {gt_path}")
    print(f"- {results_path}")
    print(f"- {summary_path}")
    print(f"- {type_summary_path}")
    print(f"- {raw_log_path}")

    print("\nSummary by question:")
    print(summary_df.to_string(index=False))

    print("\nSummary by task type:")
    print(type_summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
