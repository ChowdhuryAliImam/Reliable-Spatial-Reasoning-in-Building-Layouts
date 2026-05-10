"""
Metric graph-reasoning benchmark for Claude with real tool_use and structured output.
"""

from __future__ import annotations
from tool_call_helper import call_with_tool_use, get_api_key as helper_get_api_key
from export_utils import ensure_output_dir, export_csv

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

import connectivity_graph_data
from Tools import calculate_mean_depth, calculate_choice

# Configuration

MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
N_RUNS = int(os.getenv("N_RUNS", "5"))
WAIT_SECONDS = float(os.getenv("WAIT_SECONDS", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
MODULE_NAME = "global_tool"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", SCRIPT_DIR)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3000"))
REL_TOLERANCE = float(os.getenv("REL_TOLERANCE", "0.05"))
ABS_TOLERANCE_FOR_ZERO = float(os.getenv("ABS_TOLERANCE_FOR_ZERO", "1e-6"))
ROUND_VALUES_IN_GT = int(os.getenv("ROUND_VALUES_IN_GT", "4"))

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# API key handling

def get_api_key():
    return helper_get_api_key("Enter your Anthropic API key:")

# Graph loading

graphs: Dict[str, Dict[str, List[str]]] = {
    "graph_1": connectivity_graph_data.graph_1,
    "graph_2": connectivity_graph_data.graph_2,
    "graph_3": connectivity_graph_data.graph_3,
    "graph_4": connectivity_graph_data.graph_4,
    "graph_5": connectivity_graph_data.graph_5,
}

# Task specification

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

def sorted_unique(items):
    return sorted(set(items))

def graph_to_json(adj):
    normalized = {node: sorted_unique(neighbors) for node, neighbors in sorted(adj.items())}
    return json.dumps(normalized, indent=2)

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

def make_tasks():
    tasks: List[MetricTask] = []

    for graph_name, adj in graphs.items():
        mean_depth_raw = calculate_mean_depth(adj)
        mean_depth_gt = round_metric_dict(mean_depth_raw)
        top_3_md = rank_top_3(mean_depth_gt, mode="lowest")
        tasks.append(
            MetricTask(
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
                expected_top_3=top_3_md,
                top_3_key="top_3_lowest_mean_depth",
                ranking_rule="Lower mean depth ranks higher. Ties are broken alphabetically.",
                metadata={"metric": "mean_depth", "ranking": "lowest", "tolerance": REL_TOLERANCE},
            )
        )

        choice_raw = calculate_choice(adj)
        choice_gt = round_metric_dict(choice_raw)
        top_3_choice = rank_top_3(choice_gt, mode="highest")
        tasks.append(
            MetricTask(
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
                expected_top_3=top_3_choice,
                top_3_key="top_3_highest_choice",
                ranking_rule="Higher choice ranks higher. Ties are broken alphabetically.",
                metadata={"metric": "choice_unnormalized", "ranking": "highest", "tolerance": REL_TOLERANCE},
            )
        )

    return tasks

# Prompting

def build_prompt(task):
    output_format = {
        "metric_values": {"space_name": 0.0},
        task.top_3_key: ["space_1", "space_2", "space_3"],
    }

    return f"""
<benchmark_task>
  <role>You are evaluating a room connectivity graph using tools.</role>

  <graph_name>{task.graph_name}</graph_name>

  <question>
{task.question}
  </question>

  <instructions>
    <instruction>You must use the provided metric calculation tool first.</instruction>
    <instruction>Use graph_name exactly as provided.</instruction>
    <instruction>For mean depth tasks, call calculate_mean_depth.</instruction>
    <instruction>For choice tasks, call calculate_choice with normalized=false.</instruction>
    <instruction>After receiving tool results, rank the spaces according to the ranking rule.</instruction>
    <instruction>Then call submit_answer with the final structured answer.</instruction>
    <instruction>Do not invent spaces.</instruction>
    <instruction>Include every node returned by the metric tool in metric_values.</instruction>
    <instruction>Round metric values to 4 decimal places.</instruction>
    <instruction>{task.ranking_rule}</instruction>
  </instructions>

  <required_output_format>
{json.dumps(output_format, indent=2)}
  </required_output_format>
</benchmark_task>
""".strip()

def system_prompt():
    return (
        "You are a graph reasoning benchmark participant. "
        "You must use the provided metric tools to compute graph metrics. "
        "You must submit the final answer using submit_answer. "
        "Do not include explanations or markdown."
    )

# Claude tool schemas

def make_tools(task):
    return [
        {
            "name": "calculate_mean_depth",
            "description": "Calculate mean depth for every node in a named graph.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {
                        "type": "string",
                        "enum": list(graphs.keys()),
                    },
                },
                "required": ["graph_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "calculate_choice",
            "description": "Calculate unnormalized choice value for every node in a named graph.",
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
            "description": "Submit the final structured graph metric benchmark answer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "metric_values": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    task.top_3_key: {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                },
                "required": ["metric_values", task.top_3_key],
                "additionalProperties": False,
            },
        },
    ]

def execute_tool(tool_name, tool_input):
    graph_name = tool_input["graph_name"]
    graph = graphs[graph_name]

    if tool_name == "calculate_mean_depth":
        return calculate_mean_depth(graph)

    if tool_name == "calculate_choice":
        return calculate_choice(
            graph,
            normalized=bool(tool_input.get("normalized", False)),
        )

    raise ValueError(f"Unknown executable tool: {tool_name}")

# Claude API via requests

def call_claude(api_key, prompt, task):
    return call_with_tool_use(
        api_key=api_key,
        user_prompt=prompt,
        system_prompt=system_prompt(),
        tools=make_tools(task),
        execute_tool=execute_tool,
        model_name=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        api_url=ANTHROPIC_URL,
        anthropic_version=ANTHROPIC_VERSION,
        max_retries=MAX_RETRIES,
        wait_seconds=WAIT_SECONDS,
        submit_tool_name="submit_answer",
        return_submitted_answer=True,
    )

# Parsing and scoring

def extract_json(raw_text):
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
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

def value_within_tolerance(pred, gt):
    if abs(gt) <= ABS_TOLERANCE_FOR_ZERO:
        return abs(pred - gt) <= ABS_TOLERANCE_FOR_ZERO
    return abs(pred - gt) / abs(gt) <= REL_TOLERANCE

def metric_node_accuracy(
    predicted: Optional[Dict[str, float]],
    expected: Dict[str, float],
):
   
    if predicted is None:
        return 0.0, 0, len(expected), [], {node: None for node in expected}

    correct_nodes: List[str] = []
    rel_errors: Dict[str, Optional[float]] = {}

    for node, gt_val in expected.items():
        if node not in predicted:
            rel_errors[node] = None
            continue
        pred_val = predicted[node]
        if abs(gt_val) <= ABS_TOLERANCE_FOR_ZERO:
            rel_errors[node] = 0.0 if abs(pred_val - gt_val) <= ABS_TOLERANCE_FOR_ZERO else None
        else:
            rel_errors[node] = abs(pred_val - gt_val) / abs(gt_val)
        if value_within_tolerance(pred_val, gt_val):
            correct_nodes.append(node)

    num_correct = len(correct_nodes)
    total = len(expected)
    percent = num_correct / total if total else 0.0
    return percent, num_correct, total, correct_nodes, rel_errors

def top3_accuracy(
    predicted,
    expected,
):
    
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

    metric_acc, num_correct, total_nodes, correct_nodes, rel_errors = metric_node_accuracy(
        predicted_metric_values,
        task.expected_metric_values,
    )

    ranking_acc, ranking_num_correct, ranking_total_positions, ranking_correct_positions = top3_accuracy(
        predicted_top_3,
        task.expected_top_3,
    )

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
    rows = []
    for task in tasks:
        rows.append(
            {
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
        )
    return pd.DataFrame(rows)

def summarize_results(results_df):
    rows = []
    group_cols = [
        "graph_name",
        "task_id",
        "task_number",
        "task_type",
        "question",
        "expected_metric_values",
        "expected_top_3",
    ]

    for keys, group in results_df.groupby(group_cols, dropna=False):
        graph_name, task_id, task_number, task_type, question, expected_metric_values, expected_top_3 = keys
        metric_accuracies = group["metric_accuracy"].astype(float).tolist()
        ranking_accuracies = group["ranking_accuracy"].astype(float).tolist()
        overall_correct_values = group["overall_correct"].astype(int).tolist()

        predicted_top3_values = []
        predicted_metric_values = []
        for _, row in group.iterrows():
            try:
                predicted_top3_values.append(json.loads(row["predicted_top_3"]) if pd.notna(row["predicted_top_3"]) else None)
            except Exception:
                predicted_top3_values.append(None)
            try:
                predicted_metric_values.append(json.loads(row["predicted_metric_values"]) if pd.notna(row["predicted_metric_values"]) else None)
            except Exception:
                predicted_metric_values.append(None)

        rows.append(
            {
                "graph_name": graph_name,
                "task_id": task_id,
                "task_number": task_number,
                "task_type": task_type,
                "question": question,
                "expected_metric_values": expected_metric_values,
                "expected_top_3": expected_top_3,
                "n_runs": len(group),
                "mean_metric_accuracy": sum(metric_accuracies) / len(metric_accuracies),
                "mean_ranking_accuracy": sum(ranking_accuracies) / len(ranking_accuracies),
                "overall_exact_pass_rate": sum(overall_correct_values) / len(overall_correct_values),
                "pass_at_5_overall": int(any(v == 1 for v in overall_correct_values)),
                "pass_5_all_correct_overall": int(all(v == 1 for v in overall_correct_values)),
                "consistency_overall_correctness": binary_consistency(overall_correct_values),
                "consistency_top3_answer": consistency_score(predicted_top3_values),
                "consistency_metric_values": consistency_score(predicted_metric_values),
            }
        )

    return pd.DataFrame(rows).sort_values(["graph_name", "task_number", "task_id"])

def summarize_by_task_type(question_summary_df):
    """Aggregate only from per-question/task summary rows.

    PASS@5 and PASS^5 are first computed per question/task in
    summarize_results(), then averaged here by task_type. This avoids mixing
    raw runs with grouped task metrics.
    """
    rows = []
    for task_type, group in question_summary_df.groupby("task_type", dropna=False):
        rows.append({
            "task_type": task_type,
            "n_questions": int(len(group)),
            "mean_metric_accuracy": group["mean_metric_accuracy"].astype(float).mean(),
            "mean_ranking_accuracy": group["mean_ranking_accuracy"].astype(float).mean(),
            "overall_exact_pass_rate": group["overall_exact_pass_rate"].astype(float).mean(),
            "pass_at_5": group["pass_at_5_overall"].astype(float).mean(),
            "pass_5_all_correct": group["pass_5_all_correct_overall"].astype(float).mean(),
            "consistency_overall_correctness": group["consistency_overall_correctness"].astype(float).mean(),
            "consistency_top3_answer": group["consistency_top3_answer"].astype(float).mean(),
            "consistency_metric_values": group["consistency_metric_values"].astype(float).mean(),
        })
    return pd.DataFrame(rows).sort_values(["task_type"])

# Main

def main():
    ensure_output_dir(OUTPUT_DIR)

    tasks = make_tasks()
    gt_df = make_ground_truth_table(tasks)
    gt_path = export_csv(gt_df, OUTPUT_DIR, MODULE_NAME, "ground_truth")
    print(f"Ground truth saved to: {gt_path}")
    print(gt_df[["task_id", "question", "expected_top_3"]].to_string(index=False))

    api_key = get_api_key()
    result_rows: List[Dict[str, Any]] = []

    total_calls = len(tasks) * N_RUNS
    call_idx = 0

    for task in tasks:
        prompt = build_prompt(task)
        for run_idx in range(1, N_RUNS + 1):
            call_idx += 1
            print(f"\n[{call_idx}/{total_calls}] {task.task_id} run {run_idx}/{N_RUNS}")

            raw_answer = call_claude(api_key, prompt, task)
            parsed = extract_json(raw_answer)
            scored = score_answer(task, parsed)

            row = {
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

            partial_df = pd.DataFrame(result_rows)
            partial_path = export_csv(partial_df, OUTPUT_DIR, MODULE_NAME, "results_partial")

            print(
                "Metric accuracy: "
                f"{scored['metric_num_correct']}/{scored['metric_total_nodes']} "
                f"({scored['metric_accuracy']:.2%}) | "
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

    raw_log_df = results_df[[
        "task_id", "task_number", "task_type", "graph_name", "run_idx",
        "raw_answer", "parsed_answer", "expected_metric_values", "expected_top_3",
        "predicted_metric_values", "predicted_top_3", "metric_accuracy",
        "ranking_accuracy", "overall_correct",
    ]].copy()
    raw_log_path = export_csv(raw_log_df, OUTPUT_DIR, MODULE_NAME, "raw_answers")

    print("\nSaved outputs:")
    for path in [gt_path, results_path, summary_path, type_summary_path, raw_log_path]:
        print(f"- {path}")

    print("\nSummary by question:")
    print(summary_df.to_string(index=False))

    print("\nSummary by task type:")
    print(type_summary_df.to_string(index=False))

if __name__ == "__main__":
    main()