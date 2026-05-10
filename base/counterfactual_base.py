from __future__ import annotations

import os
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import connectivity_graph_data
import Tools
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

MODULE_NAME = "counterfactual_base"
CONFIG = LLMConfig.from_env(default_output_dir="benchmark_outputs", default_max_tokens=500, default_n_runs=5)

MODEL_NAME = CONFIG.model_name
N_RUNS = CONFIG.n_runs
WAIT_SECONDS = CONFIG.wait_seconds
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", SCRIPT_DIR)
TEMPERATURE = CONFIG.temperature
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

def add_new_bedroom(adj, attach):
    graph = deepcopy(adj)
    if attach not in graph:
        raise ValueError(f"Attach node not found in graph: {attach}")
    if NEW_NODE not in graph[attach]:
        graph[attach].append(NEW_NODE)
    graph[NEW_NODE] = [attach]
    return graph

def mean_depth(adj):
    return Tools.calculate_mean_depth(adj)

def choice(adj):
    return Tools.calculate_choice(adj, normalized=False)

def compute_ground_truth():
    gt = {}
    for name, graph in graphs.items():
        cf_graph = add_new_bedroom(graph, ATTACH_TO[name])
        gt[name] = {
            "Task_6_counterfactual_mean_depth": {
                "changed_top_3_lowest_mean_depth": rank_top_k(mean_depth(cf_graph), k=3, mode="lowest")
            },
            "Task_7_counterfactual_choice": {
                "changed_top_3_highest_choice": rank_top_k(choice(cf_graph), k=3, mode="highest")
            },
        }
    return gt

def make_tasks(gt):
    tasks: List[Task] = []
    for graph_name in graphs.keys():
        attach = ATTACH_TO[graph_name]
        tasks.append(Task(
            task_id=f"{graph_name}_T6",
            task_type="Task_6_counterfactual_mean_depth",
            graph_name=graph_name,
            question=f"After adding a new bedroom to {attach}, return top 3 lowest mean depth spaces.",
            expected_answer=gt[graph_name]["Task_6_counterfactual_mean_depth"],
        ))
        tasks.append(Task(
            task_id=f"{graph_name}_T7",
            task_type="Task_7_counterfactual_choice",
            graph_name=graph_name,
            question=f"After adding a new bedroom to {attach}, return top 3 highest choice spaces.",
            expected_answer=gt[graph_name]["Task_7_counterfactual_choice"],
        ))
    return tasks

SYSTEM_PROMPT = """
You are a counterfactual graph-reasoning benchmark participant.
Use only the provided graph and requested counterfactual edit.
You must submit the final answer using the structured submit_answer tool.
Do not include explanations, markdown, or prose outside the structured answer.
""".strip()

def build_prompt(task):
    graph = graphs[task.graph_name]
    attach = ATTACH_TO[task.graph_name]
    key = answer_key(task)
    output_format = {key: ["space_1", "space_2", "space_3"]}
    metric_name = "mean depth" if "mean_depth" in task.task_type else "unnormalized choice"
    ranking_rule = (
        "Return the top 3 spaces with the lowest mean depth. Ties are broken alphabetically."
        if "mean_depth" in task.task_type
        else "Return the top 3 spaces with the highest unnormalized choice. Ties are broken alphabetically."
    )
    return f"""
<benchmark_task>
  <role>You are evaluating a room connectivity graph. Treat the graph as undirected.</role>
  <graph_name>{task.graph_name}</graph_name>
  <connectivity_graph>
{graph_to_json(graph)}
  </connectivity_graph>
  <counterfactual_change>Add {NEW_NODE} connected to {attach}.</counterfactual_change>
  <question>{task.question}</question>
  <instructions>
    <instruction>Apply the counterfactual change before reasoning.</instruction>
    <instruction>Compute {metric_name} after the edit.</instruction>
    <instruction>{ranking_rule}</instruction>
    <instruction>Use exact node names from the graph.</instruction>
    <instruction>Return the final answer only through submit_answer.</instruction>
  </instructions>
  <required_output_format>
{json.dumps(output_format, indent=2)}
  </required_output_format>
</benchmark_task>
""".strip()

def expected_top3(task):
    return list(task.expected_answer.values())[0]

def answer_key(task):
    return "changed_top_3_lowest_mean_depth" if "mean_depth" in task.task_type else "changed_top_3_highest_choice"

def answer_schema(task):
    key = answer_key(task)
    return {
        "type": "object",
        "properties": {
            key: {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
            }
        },
        "required": [key],
        "additionalProperties": False,
    }

def parse(text, task):
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        return None
    value = parsed.get(answer_key(task))
    if not isinstance(value, list):
        return None
    return [str(x).strip() for x in value]

def score_answer(task, parsed):
    expected = expected_top3(task)
    ranking_acc, num_correct, total_positions, correct_positions = top3_accuracy(parsed, expected)
    return {
        "predicted_top_3": parsed,
        "ranking_accuracy": ranking_acc,
        "ranking_num_correct": num_correct,
        "ranking_total_positions": total_positions,
        "ranking_correct_positions": correct_positions,
        "overall_correct": int(parsed == expected),
    }

def make_ground_truth_table(tasks):
    return pd.DataFrame([
        {
            "task_id": task.task_id,
            "graph_name": task.graph_name,
            "task_type": task.task_type,
            "question": task.question,
            "expected_top_3": json_dumps(expected_top3(task)),
            "expected_answer": json_dumps(task.expected_answer),
        }
        for task in tasks
    ])

def summarize_results(results_df):
    rows = []
    group_cols = ["graph_name", "task_id", "task_type", "question", "expected_top_3"]
    for keys, group in results_df.groupby(group_cols, dropna=False):
        graph_name, task_id, task_type, question, expected_top_3 = keys
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
            "mean_ranking_accuracy": group["ranking_accuracy"].astype(float).mean(),
            "overall_exact_pass_rate": group["overall_correct"].mean(),
            "pass_at_5_overall": int(any(v == 1 for v in overall_correct_values)),
            "pass_5_all_correct_overall": int(all(v == 1 for v in overall_correct_values)),
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

def main():
    api_key = get_api_key()
    gt = compute_ground_truth()
    tasks = make_tasks(gt)
    gt_df = make_ground_truth_table(tasks)
    gt_path = export_csv(gt_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="ground_truth")
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
            raw_answer = call_claude(api_key=api_key, user_prompt=prompt, system_prompt=SYSTEM_PROMPT, config=CONFIG, answer_schema=answer_schema(task))
            parsed = parse(raw_answer, task)
            scored = score_answer(task, parsed)
            expected = expected_top3(task)
            row = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "graph_name": task.graph_name,
                "run_idx": run_idx,
                "question": task.question,
                "expected_top_3": json_dumps(expected),
                "raw_answer": raw_answer,
                "parsed_answer": json_dumps(parsed) if parsed is not None else None,
                "predicted_top_3": json_dumps(scored["predicted_top_3"]),
                "ranking_accuracy": scored["ranking_accuracy"],
                "ranking_num_correct": scored["ranking_num_correct"],
                "ranking_total_positions": scored["ranking_total_positions"],
                "ranking_correct_positions": json_dumps(scored["ranking_correct_positions"]),
                "overall_correct": scored["overall_correct"],
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
            }
            result_rows.append(row)
            export_csv(pd.DataFrame(result_rows), output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="results_partial")
            print("Raw answer:")
            print(raw_answer)
            print(f"Ranking accuracy: {scored['ranking_num_correct']}/{scored['ranking_total_positions']} ({scored['ranking_accuracy']:.2%}) | Overall: {scored['overall_correct']}")
            print(f"Expected top 3: {expected}")
            print(f"Predicted top 3: {scored['predicted_top_3']}")
            pause_between_runs(WAIT_SECONDS)

    results_df = pd.DataFrame(result_rows)
    results_path = export_csv(results_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="results")
    summary_df = summarize_results(results_df)
    summary_path = export_csv(summary_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="summary_by_question")
    type_summary_df = summarize_by_task_type(summary_df)
    type_summary_path = export_csv(type_summary_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="summary_by_type")
    raw_log_path = export_csv(results_df[["task_id", "task_type", "graph_name", "run_idx", "raw_answer", "parsed_answer", "expected_top_3", "predicted_top_3", "ranking_accuracy", "overall_correct"]], output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="raw_answers")

    print("\nSaved outputs:")
    for path in [gt_path, results_path, summary_path, type_summary_path, raw_log_path]:
        print(f"- {path}")
    print("\nSummary by question:")
    print(summary_df.to_string(index=False))
    print("\nSummary by task type:")
    print(type_summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
