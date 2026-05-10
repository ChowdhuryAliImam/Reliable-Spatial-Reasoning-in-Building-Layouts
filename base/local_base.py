from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

import connectivity_graph_data
from base_helper import (
    LLMConfig,
    binary_consistency,
    call_claude,
    extract_json,
    get_api_key,
    graph_to_json,
    json_dumps,
    pause_between_runs,
    sorted_unique,
)
from export_utils import export_csv

MODULE_NAME = "local_base"
CONFIG = LLMConfig.from_env(default_output_dir="benchmark_outputs", default_max_tokens=700)

MODEL_NAME = CONFIG.model_name
N_RUNS = CONFIG.n_runs
WAIT_SECONDS = CONFIG.wait_seconds
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", SCRIPT_DIR)
TEMPERATURE = CONFIG.temperature

graphs: Dict[str, Dict[str, List[str]]] = {
    "graph_1": connectivity_graph_data.graph_1,
    "graph_2": connectivity_graph_data.graph_2,
    "graph_3": connectivity_graph_data.graph_3,
    "graph_4": connectivity_graph_data.graph_4,
    "graph_5": connectivity_graph_data.graph_5,
}

L1_TARGETS = {"graph_1": "hall", "graph_2": "hall", "graph_3": "hall", "graph_4": "hall", "graph_5": "hall_1"}
L2_PAIRS = {
    "graph_1": ("living_room", "dining_room"),
    "graph_2": ("living_room", "dining_room"),
    "graph_3": ("living_room", "dining_room"),
    "graph_4": ("living_room", "dining_room"),
    "graph_5": ("living_room_1", "dining_room_1"),
}
L3_TARGETS = {"graph_1": "hall", "graph_2": "hall", "graph_3": "hall", "graph_4": "hall", "graph_5": "hall_1"}

@dataclass
class Task:
    task_id: str
    task_type: str
    graph_name: str
    question: str
    expected_answer: Any
    answer_format: str
    metadata: Dict[str, Any]

def build_nx_graph(adj_list):
    graph = nx.Graph()
    for node, neighbors in adj_list.items():
        graph.add_node(node)
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)
    return graph

def neighbors_of(graph, node):
    return sorted_unique(list(graph.neighbors(node)))

def degree_winner(graph, a, b):
    deg_a = int(graph.degree[a])
    deg_b = int(graph.degree[b])
    if deg_a > deg_b:
        winner = a
    elif deg_b > deg_a:
        winner = b
    else:
        winner = "tie"
    return {"winner": winner, "degree_a": deg_a, "degree_b": deg_b}

def nodes_within_two_steps(graph, source, include_source = False):
    lengths = nx.single_source_shortest_path_length(graph, source, cutoff=2)
    nodes = [node for node, dist in lengths.items() if dist <= 2]
    if not include_source:
        nodes = [node for node in nodes if node != source]
    return sorted_unique(nodes)

def make_tasks():
    tasks: List[Task] = []
    for graph_name, adj in graphs.items():
        graph = build_nx_graph(adj)

        x = L1_TARGETS[graph_name]
        tasks.append(Task(
            task_id=f"{graph_name}_L1",
            task_type="L1_direct_adjacency",
            graph_name=graph_name,
            question=f"Which spaces are directly connected to {x}?",
            expected_answer=neighbors_of(graph, x),
            answer_format='{"answer": ["space_1", "space_2"]}',
            metadata={"room_x": x},
        ))

        a, b = L2_PAIRS[graph_name]
        tasks.append(Task(
            task_id=f"{graph_name}_L2",
            task_type="L2_degree_comparison",
            graph_name=graph_name,
            question=f"Which space has more direct connections: {a} or {b}?",
            expected_answer=degree_winner(graph, a, b),
            answer_format='{"answer": "space_name_or_tie"}',
            metadata={"room_a": a, "room_b": b},
        ))

        x = L3_TARGETS[graph_name]
        tasks.append(Task(
            task_id=f"{graph_name}_L3",
            task_type="L3_two_step_reachability",
            graph_name=graph_name,
            question=f"Which spaces can be reached from {x} in two steps or fewer? Exclude {x} itself.",
            expected_answer=nodes_within_two_steps(graph, x, include_source=False),
            answer_format='{"answer": ["space_1", "space_2"]}',
            metadata={"room_x": x, "include_source": False},
        ))
    return tasks

def build_prompt(task):
    graph_json = graph_to_json(graphs[task.graph_name])
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
    <instruction>Return JSON only. Do not include markdown, prose, or explanation.</instruction>
    <instruction>For list answers, include all correct spaces and sort them alphabetically.</instruction>
  </instructions>
  <required_output_format>{task.answer_format}</required_output_format>
</benchmark_task>
""".strip()

def build_system_prompt():
    return (
        "You are a graph-reasoning benchmark participant. "
        "Answer only from the provided connectivity graph. "
        "Use exact node names. Return JSON only."
    )

def normalize_list_answer(value):
    if not isinstance(value, list):
        return None
    return sorted_unique([str(x).strip() for x in value])

def normalize_scalar_answer(value):
    if value is None:
        return None
    return str(value).strip()

def score_answer(task, parsed):
    if parsed is None or "answer" not in parsed:
        return 0, None

    if task.task_type in {"L1_direct_adjacency", "L3_two_step_reachability"}:
        model_answer = normalize_list_answer(parsed.get("answer"))
        expected = sorted_unique(task.expected_answer)
        return int(model_answer == expected), model_answer

    if task.task_type == "L2_degree_comparison":
        model_answer = normalize_scalar_answer(parsed.get("answer"))
        expected_winner = task.expected_answer["winner"]
        return int(model_answer == expected_winner), model_answer

    raise ValueError(f"Unknown task type: {task.task_type}")

def make_ground_truth_table(tasks):
    return pd.DataFrame([
        {
            "task_id": task.task_id,
            "graph_name": task.graph_name,
            "task_type": task.task_type,
            "question": task.question,
            "expected_answer": json_dumps(task.expected_answer),
            "metadata": json_dumps(task.metadata),
        }
        for task in tasks
    ])

def summarize_results(results_df):
    rows = []
    group_cols = ["graph_name", "task_id", "task_type", "question", "expected_answer"]
    for keys, group in results_df.groupby(group_cols, dropna=False):
        graph_name, task_id, task_type, question, expected_answer = keys
        correct_values = group["correct"].astype(int).tolist()
        rows.append({
            "graph_name": graph_name,
            "task_id": task_id,
            "task_type": task_type,
            "question": question,
            "expected_answer": expected_answer,
            "n_runs": len(correct_values),
            "accuracy": sum(correct_values) / len(correct_values),
            "pass_at_5": int(any(v == 1 for v in correct_values)),
            "pass_5_all_correct": int(all(v == 1 for v in correct_values)),
            "consistency_majority_correctness": binary_consistency(correct_values),
        })
    return pd.DataFrame(rows).sort_values(["graph_name", "task_id"])

def summarize_by_task_type(question_summary_df):
    rows = []
    for task_type, group in question_summary_df.groupby("task_type", dropna=False):
        rows.append({
            "task_type": task_type,
            "n_questions": int(len(group)),
            "accuracy": group["accuracy"].astype(float).mean(),
            "pass_at_5": group["pass_at_5"].astype(float).mean(),
            "pass_5_all_correct": group["pass_5_all_correct"].astype(float).mean(),
            "consistency_majority_correctness": group["consistency_majority_correctness"].astype(float).mean(),
        })
    return pd.DataFrame(rows).sort_values(["task_type"])

def answer_schema(task):
 
    if task.task_type in {"L1_direct_adjacency", "L3_two_step_reachability"}:
        answer_field: Dict[str, Any] = {"type": "array", "items": {"type": "string"}}
    else:
        answer_field = {"type": "string"}
    return {
        "type": "object",
        "properties": {"answer": answer_field},
        "required": ["answer"],
        "additionalProperties": False,
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tasks = make_tasks()
    gt_df = make_ground_truth_table(tasks)
    gt_path = export_csv(gt_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="ground_truth")
    print(f"Ground truth saved to: {gt_path}")
    print(gt_df[["task_id", "question", "expected_answer"]].to_string(index=False))

    api_key = get_api_key()
    system_prompt = build_system_prompt()
    result_rows: List[Dict[str, Any]] = []
    total_calls = len(tasks) * N_RUNS
    call_idx = 0

    for task in tasks:
        prompt = build_prompt(task)
        for run_idx in range(1, N_RUNS + 1):
            call_idx += 1
            print(f"\n[{call_idx}/{total_calls}] {task.task_id} run {run_idx}/{N_RUNS}")
            raw_answer = call_claude(api_key=api_key, user_prompt=prompt, system_prompt=system_prompt, config=CONFIG, answer_schema=answer_schema(task))
            parsed = extract_json(raw_answer)
            correct, normalized_model_answer = score_answer(task, parsed)
            row = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "graph_name": task.graph_name,
                "run_idx": run_idx,
                "question": task.question,
                "expected_answer": json_dumps(task.expected_answer),
                "raw_answer": raw_answer,
                "parsed_answer": json_dumps(parsed) if parsed is not None else None,
                "normalized_model_answer": json_dumps(normalized_model_answer),
                "correct": correct,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
            }
            result_rows.append(row)
            partial_df = pd.DataFrame(result_rows)
            export_csv(partial_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="results_partial")
            print(f"Correct: {correct} | Answer: {normalized_model_answer}")
            pause_between_runs(WAIT_SECONDS)

    results_df = pd.DataFrame(result_rows)
    results_path = export_csv(results_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="results")
    summary_df = summarize_results(results_df)
    summary_path = export_csv(summary_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="summary_by_question")
    type_summary_df = summarize_by_task_type(summary_df)
    type_summary_path = export_csv(type_summary_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="summary_by_type")

    raw_log_path = export_csv(
        results_df[[
            "task_id", "task_type", "graph_name", "run_idx", "raw_answer",
            "parsed_answer", "expected_answer", "normalized_model_answer", "correct"
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
