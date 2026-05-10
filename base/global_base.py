from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import connectivity_graph_data
from Tools import calculate_choice, calculate_mean_depth
from base_helper import (
    LLMConfig,
    binary_consistency,
    call_claude,
    consistency_score,
    extract_json,
    get_api_key,
    graph_to_json,
    json_dumps,
    pause_between_runs,
    rank_top_k,
    top3_accuracy,
)
from export_utils import export_csv

MODULE_NAME = "global_base"
CONFIG = LLMConfig.from_env(default_output_dir="benchmark_outputs", default_max_tokens=1400)

MODEL_NAME = CONFIG.model_name
N_RUNS = CONFIG.n_runs
WAIT_SECONDS = CONFIG.wait_seconds
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", SCRIPT_DIR)
TEMPERATURE = CONFIG.temperature
REL_TOLERANCE = float(__import__("os").getenv("REL_TOLERANCE", "0.05"))
ABS_TOLERANCE_FOR_ZERO = float(__import__("os").getenv("ABS_TOLERANCE_FOR_ZERO", "1e-6"))
ROUND_VALUES_IN_GT = int(__import__("os").getenv("ROUND_VALUES_IN_GT", "4"))

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

def round_metric_dict(values, digits = ROUND_VALUES_IN_GT):
    return {str(k): round(float(v), digits) for k, v in sorted(values.items())}

def make_tasks():
    tasks: List[MetricTask] = []
    for graph_name, adj in graphs.items():
        mean_depth_gt = round_metric_dict(calculate_mean_depth(adj))
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
            expected_top_3=rank_top_k(mean_depth_gt, k=3, mode="lowest"),
            top_3_key="top_3_lowest_mean_depth",
            ranking_rule="Lower mean depth ranks higher. Ties are broken alphabetically.",
            metadata={"metric": "mean_depth", "ranking": "lowest", "tolerance": REL_TOLERANCE},
        ))

        choice_gt = round_metric_dict(calculate_choice(adj))
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
            expected_top_3=rank_top_k(choice_gt, k=3, mode="highest"),
            top_3_key="top_3_highest_choice",
            ranking_rule="Higher choice ranks higher. Ties are broken alphabetically.",
            metadata={"metric": "choice_unnormalized", "ranking": "highest", "tolerance": REL_TOLERANCE},
        ))
    return tasks

def build_prompt(task):
    graph_json = graph_to_json(graphs[task.graph_name])
    output_format = {
        "metric_values": {"space_name": 0.0},
        task.top_3_key: ["space_1", "space_2", "space_3"],
    }
    return f"""
<benchmark_task>
  <role>You are evaluating a room connectivity graph. Treat the graph as undirected.</role>
  <graph_name>{task.graph_name}</graph_name>
  <connectivity_graph>
{graph_json}
  </connectivity_graph>
  <question>{task.question}</question>
  <instructions>
    <instruction>Answer only from the graph.</instruction>
    <instruction>Do not invent spaces.</instruction>
    <instruction>Use exact node names from the graph.</instruction>
    <instruction>Include every node in metric_values.</instruction>
    <instruction>Round metric values to 4 decimal places.</instruction>
    <instruction>{task.ranking_rule}</instruction>
    <instruction>Return JSON only. Do not include markdown, prose, or explanation.</instruction>
  </instructions>
  <required_output_format>
{json.dumps(output_format, indent=2)}
  </required_output_format>
</benchmark_task>
""".strip()

def system_prompt():
    return (
        "You are a graph reasoning benchmark participant. "
        "You must follow the user's required JSON output format exactly. "
        "Do not include explanations or markdown."
    )

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

def metric_node_accuracy(predicted, expected):
    if predicted is None:
        return 0.0, 0, len(expected), [], {node: None for node in expected}
    correct_nodes: List[str] = []
    rel_errors: Dict[str, Optional[float]] = {}
    for node, gt_val in expected.items():
        if node not in predicted:
            rel_errors[node] = None
            continue
        pred_val = predicted[node]
        rel_errors[node] = 0.0 if abs(gt_val) <= ABS_TOLERANCE_FOR_ZERO and abs(pred_val - gt_val) <= ABS_TOLERANCE_FOR_ZERO else (
            None if abs(gt_val) <= ABS_TOLERANCE_FOR_ZERO else abs(pred_val - gt_val) / abs(gt_val)
        )
        if value_within_tolerance(pred_val, gt_val):
            correct_nodes.append(node)
    total = len(expected)
    num_correct = len(correct_nodes)
    return (num_correct / total if total else 0.0), num_correct, total, correct_nodes, rel_errors

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

def make_ground_truth_table(tasks):
    return pd.DataFrame([
        {
            "task_id": task.task_id,
            "task_number": task.task_number,
            "graph_name": task.graph_name,
            "task_type": task.task_type,
            "question": task.question,
            "expected_metric_values": json_dumps(task.expected_metric_values),
            "expected_top_3": json_dumps(task.expected_top_3),
            "top_3_key": task.top_3_key,
            "ranking_rule": task.ranking_rule,
            "metadata": json_dumps(task.metadata),
        }
        for task in tasks
    ])

def summarize_results(results_df):
    rows = []
    group_cols = ["graph_name", "task_id", "task_number", "task_type", "question", "expected_metric_values", "expected_top_3"]
    for keys, group in results_df.groupby(group_cols, dropna=False):
        graph_name, task_id, task_number, task_type, question, expected_metric_values, expected_top_3 = keys
        overall_correct_values = group["overall_correct"].astype(int).tolist()
        predicted_top3_values = [json.loads(x) if pd.notna(x) else None for x in group["predicted_top_3"]]
        predicted_metric_values = [json.loads(x) if pd.notna(x) else None for x in group["predicted_metric_values"]]
        rows.append({
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
            "task_type": task_type,
            "n_questions": int(len(group)),
            "mean_metric_accuracy": group["mean_metric_accuracy"].astype(float).mean(),
            "mean_ranking_accuracy": group["mean_ranking_accuracy"].astype(float).mean(),
            "pass_at_5": group["pass_at_5_overall"].astype(float).mean(),
            "pass_5_all_correct": group["pass_5_all_correct_overall"].astype(float).mean(),
            "consistency_overall_correctness": group["consistency_overall_correctness"].astype(float).mean(),
            "consistency_top3_answer": group["consistency_top3_answer"].astype(float).mean(),
            "consistency_metric_values": group["consistency_metric_values"].astype(float).mean(),
        })
    return pd.DataFrame(rows).sort_values(["task_type"])

def answer_schema(task):
    return {
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
    }

def main():
    tasks = make_tasks()
    gt_df = make_ground_truth_table(tasks)
    gt_path = export_csv(gt_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="ground_truth")
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
            raw_answer = call_claude(api_key=api_key, user_prompt=prompt, system_prompt=system_prompt(), config=CONFIG, answer_schema=answer_schema(task))
            parsed = extract_json(raw_answer)
            scored = score_answer(task, parsed)
            row = {
                "task_id": task.task_id,
                "task_number": task.task_number,
                "task_type": task.task_type,
                "graph_name": task.graph_name,
                "run_idx": run_idx,
                "question": task.question,
                "expected_metric_values": json_dumps(task.expected_metric_values),
                "expected_top_3": json_dumps(task.expected_top_3),
                "raw_answer": raw_answer,
                "parsed_answer": json_dumps(parsed) if parsed is not None else None,
                "predicted_metric_values": json_dumps(scored["predicted_metric_values"]),
                "predicted_top_3": json_dumps(scored["predicted_top_3"]),
                "metric_accuracy": scored["metric_accuracy"],
                "metric_num_correct": scored["metric_num_correct"],
                "metric_total_nodes": scored["metric_total_nodes"],
                "metric_correct_nodes": json_dumps(scored["metric_correct_nodes"]),
                "ranking_accuracy": scored["ranking_accuracy"],
                "ranking_num_correct": scored["ranking_num_correct"],
                "ranking_total_positions": scored["ranking_total_positions"],
                "ranking_correct_positions": json_dumps(scored["ranking_correct_positions"]),
                "overall_correct": scored["overall_correct"],
                "per_node_relative_errors": json_dumps(scored["per_node_relative_errors"]),
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "rel_tolerance": REL_TOLERANCE,
            }
            result_rows.append(row)
            export_csv(pd.DataFrame(result_rows), output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="results_partial")
            print(
                f"Metric accuracy: {scored['metric_num_correct']}/{scored['metric_total_nodes']} ({scored['metric_accuracy']:.2%}) | "
                f"Ranking accuracy: {scored['ranking_num_correct']}/{scored['ranking_total_positions']} ({scored['ranking_accuracy']:.2%}) | "
                f"Overall: {scored['overall_correct']}"
            )
            print(f"Predicted top 3: {scored['predicted_top_3']}")
            pause_between_runs(WAIT_SECONDS)

    results_df = pd.DataFrame(result_rows)
    results_path = export_csv(results_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="results")
    summary_df = summarize_results(results_df)
    summary_path = export_csv(summary_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="summary_by_question")
    type_summary_df = summarize_by_task_type(summary_df)
    type_summary_path = export_csv(type_summary_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="summary_by_type")
    raw_log_path = export_csv(
        results_df[[
            "task_id", "task_number", "task_type", "graph_name", "run_idx",
            "raw_answer", "parsed_answer", "expected_top_3", "predicted_top_3",
            "metric_accuracy", "ranking_accuracy", "overall_correct"
        ]],
        output_dir=OUTPUT_DIR,
        module_name=MODULE_NAME,
        artifact_name="raw_answers",
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
