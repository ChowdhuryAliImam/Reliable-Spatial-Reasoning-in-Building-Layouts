from __future__ import annotations

import os
import json
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
)
from export_utils import export_csv

MODULE_NAME = "conceptual_base"
CONFIG = LLMConfig.from_env(default_output_dir=".", default_max_tokens=500, default_n_runs=1)

MODEL_NAME = CONFIG.model_name
N_RUNS = CONFIG.n_runs
WAIT_SECONDS = CONFIG.wait_seconds
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", SCRIPT_DIR)
TEMPERATURE = CONFIG.temperature
SEED = int(os.getenv("SEED", "42"))

graphs: Dict[str, Dict[str, List[str]]] = {
    "graph_1": connectivity_graph_data.graph_1,
    "graph_3": connectivity_graph_data.graph_3,
    "graph_4": connectivity_graph_data.graph_4,
}

@dataclass
class Task:
    task_id: str
    task_type: str
    pair_id: str
    graph_A: str
    graph_B: str
    question: str
    expected_answer: Dict[str, Any]
    metadata: Dict[str, Any]

def is_entry(node):
    node_l = node.lower()
    return ("entry" in node_l) or ("porch" in node_l and "sleeping" not in node_l)

def is_private(node):
    node_l = node.lower()
    return node_l.startswith("bed_room") or node_l.startswith("bath_room") or node_l.startswith("sleeping_porch")

def is_shared_or_guest_facing(node):
    node_l = node.lower()
    return node_l.startswith("living_room") or node_l.startswith("dining_room") or node_l.startswith("hall")

def valid_entries(adj):
    return sorted([node for node in adj if is_entry(node)])

def private_spaces(adj):
    return sorted([node for node in adj if is_private(node)])

def non_entry_spaces(adj):
    return sorted([node for node in adj if not is_entry(node)])

def call_tool_choice(adj):
    try:
        return Tools.calculate_choice(adj, normalized=False)
    except TypeError:
        return Tools.calculate_choice(adj)

def call_tool_shortest_path_info(adj, source, target):
    return Tools.calculate_shortest_path_info(adj, source, target)

def call_tool_shortest_distance(adj, source, target):
    return float(call_tool_shortest_path_info(adj, source, target)["distance"])

def average_nearest_entry_distance(adj, targets):
    entries = valid_entries(adj)
    if not entries:
        raise ValueError("No valid entry spaces found.")
    if not targets:
        raise ValueError("No target spaces found for distance calculation.")
    distances: List[float] = []
    for target in targets:
        candidate_entries = [entry for entry in entries if entry != target]
        if not candidate_entries:
            distances.append(0.0)
            continue
        distances.append(min(call_tool_shortest_distance(adj, entry, target) for entry in candidate_entries))
    return sum(distances) / len(distances)

def private_exposure_score(adj):
    entries = valid_entries(adj)
    privates = private_spaces(adj)
    if not entries:
        raise ValueError("No valid entry spaces found.")
    if not privates:
        raise ValueError("No private spaces found.")
    exposure_counts: List[float] = []
    for private in privates:
        best_path: Optional[List[str]] = None
        best_distance = float("inf")
        for entry in entries:
            info = call_tool_shortest_path_info(adj, entry, private)
            path = info["path"]
            distance = float(info["distance"])
            if distance < best_distance:
                best_distance = distance
                best_path = path
        if best_path is None:
            raise RuntimeError(f"No path found from entries {entries} to private space {private}.")
        intermediate_nodes = best_path[1:-1]
        exposure_counts.append(float(sum(1 for node in intermediate_nodes if is_shared_or_guest_facing(node))))
    return sum(exposure_counts) / len(exposure_counts)

def bottleneck_score(adj):
    choice_values = call_tool_choice(adj)
    return max(float(v) for v in choice_values.values())

def entry_access_score(adj):
    return average_nearest_entry_distance(adj, non_entry_spaces(adj))

def compare_scores(value_A, value_B, higher_is_better):
    if value_A == value_B:
        return "tie"
    if higher_is_better:
        return "A" if value_A > value_B else "B"
    return "A" if value_A < value_B else "B"

def generate_pairs():
    return [("graph_1", "graph_4"), ("graph_4", "graph_3"), ("graph_3", "graph_1")]

PARAPHRASES: Dict[str, List[str]] = {
    "S1_private_space_exposure": [
        "Between Layout A and Layout B, which layout better protects private spaces from exposure through shared or guest-facing spaces?",
        "Between Layout A and Layout B, which layout gives private rooms less exposure to shared circulation or guest-facing areas?",
        "Between Layout A and Layout B, which layout better avoids routing access to private spaces through shared spaces?",
        "Between Layout A and Layout B, which layout keeps private spaces more separated from guest-facing paths?",
        "Between Layout A and Layout B, which layout reduces the number of shared spaces someone passes through before reaching private rooms?",
    ],
    "S2_bottleneck_reduction": [
        "Between Layout A and Layout B, which layout is more efficient at reducing bottleneck effects in circulation?",
        "Between Layout A and Layout B, which layout avoids concentrating movement through a single space more effectively?",
        "Between Layout A and Layout B, which layout distributes movement more evenly across the plan?",
        "Between Layout A and Layout B, which layout is less likely to create a circulation bottleneck?",
        "Between Layout A and Layout B, which layout avoids over-reliance on one space for movement?",
    ],
    "S3_entry_access_efficiency": [
        "Between Layout A and Layout B, which layout provides better access from valid entry spaces to the rest of the layout?",
        "Between Layout A and Layout B, which layout places entries more efficiently for reaching rooms?",
        "Between Layout A and Layout B, which layout provides shorter overall access from entries to spaces?",
        "Between Layout A and Layout B, which layout gives more efficient entry-to-room access?",
        "Between Layout A and Layout B, which layout has better overall accessibility from entry points?",
    ],
}

TASK_LABELS = {
    "S1_private_space_exposure": "S1",
    "S2_bottleneck_reduction": "S2",
    "S3_entry_access_efficiency": "S3",
}

def compute_ground_truth_for_pair(task_type, graph_A, graph_B):
    adj_A = graphs[graph_A]
    adj_B = graphs[graph_B]
    if task_type == "S1_private_space_exposure":
        score_A = private_exposure_score(adj_A)
        score_B = private_exposure_score(adj_B)
        return {
            "answer": compare_scores(score_A, score_B, higher_is_better=False),
            "criterion": "lower average shared/guest-facing exposure on entry-to-private paths",
            "layout_A_score": score_A,
            "layout_B_score": score_B,
            "higher_is_better": False,
            "layout_A_entries": valid_entries(adj_A),
            "layout_B_entries": valid_entries(adj_B),
            "layout_A_private_spaces": private_spaces(adj_A),
            "layout_B_private_spaces": private_spaces(adj_B),
        }
    if task_type == "S2_bottleneck_reduction":
        score_A = bottleneck_score(adj_A)
        score_B = bottleneck_score(adj_B)
        return {
            "answer": compare_scores(score_A, score_B, higher_is_better=False),
            "criterion": "lower maximum unnormalized choice value",
            "layout_A_score": score_A,
            "layout_B_score": score_B,
            "higher_is_better": False,
        }
    if task_type == "S3_entry_access_efficiency":
        score_A = entry_access_score(adj_A)
        score_B = entry_access_score(adj_B)
        return {
            "answer": compare_scores(score_A, score_B, higher_is_better=False),
            "criterion": "lower average nearest-entry distance to all non-entry spaces",
            "layout_A_score": score_A,
            "layout_B_score": score_B,
            "higher_is_better": False,
            "layout_A_entries": valid_entries(adj_A),
            "layout_B_entries": valid_entries(adj_B),
        }
    raise ValueError(f"Unknown task type: {task_type}")

def make_tasks():
    tasks: List[Task] = []
    pairs = generate_pairs()
    task_to_pair = {
        "S1_private_space_exposure": pairs[0],
        "S2_bottleneck_reduction": pairs[1],
        "S3_entry_access_efficiency": pairs[2],
    }
    for task_type, questions in PARAPHRASES.items():
        task_short = TASK_LABELS[task_type]
        graph_A, graph_B = task_to_pair[task_type]
        expected = compute_ground_truth_for_pair(task_type, graph_A, graph_B)
        for paraphrase_idx, question in enumerate(questions, start=1):
            tasks.append(Task(
                task_id=f"{task_short}_p{paraphrase_idx}",
                task_type=task_type,
                pair_id=f"{task_short}_pair",
                graph_A=graph_A,
                graph_B=graph_B,
                question=question,
                expected_answer=expected,
                metadata={"paraphrase_index": paraphrase_idx, "seed": SEED, "graph_A": graph_A, "graph_B": graph_B},
            ))
    return tasks

SYSTEM_PROMPT = """
You are a semantic design-reasoning benchmark participant.
Choose exactly one option: A or B.
Return JSON only with exactly these fields: answer and criterion_used.
Do not include markdown, prose, or explanation outside JSON.
""".strip()

def semantic_context_for_prompt(task):
    base_context = """
Semantic definitions for this benchmark:
- Valid entries are any node containing "entry" or any porch, except sleeping_porch.
- Private spaces are bed_room*, bath_room*, and sleeping_porch*.
- Shared or guest-facing spaces are living_room*, dining_room*, and hall*.
""".strip()
    if task.task_type == "S1_private_space_exposure":
        return base_context + "\n\nTask interpretation:\n- Better private-space protection means private spaces are reached with fewer passes through shared or guest-facing spaces."
    if task.task_type == "S2_bottleneck_reduction":
        return base_context + "\n\nTask interpretation:\n- A bottleneck means movement is overly concentrated through one space.\n- The better layout is the one with weaker bottleneck effect."
    if task.task_type == "S3_entry_access_efficiency":
        return base_context + "\n\nTask interpretation:\n- Better entry access means rooms are more efficiently reachable from valid entry spaces."
    raise ValueError(f"Unknown task type: {task.task_type}")

def build_prompt(task):
    return f"""
Layout A ({task.graph_A}):
{graph_to_json(graphs[task.graph_A])}

Layout B ({task.graph_B}):
{graph_to_json(graphs[task.graph_B])}

{semantic_context_for_prompt(task)}

Question:
{task.question}

Use the submit_answer structured-output tool in this exact shape:
{{"answer": "A", "criterion_used": "short phrase describing the criterion used"}}
""".strip()

def parse(raw_text):
    parsed_json = extract_json(raw_text)
    if not isinstance(parsed_json, dict):
        return None
    answer = parsed_json.get("answer")
    if isinstance(answer, str):
        answer = answer.strip().upper()
    if answer not in {"A", "B"}:
        return None
    return {"answer": answer, "criterion_used": str(parsed_json.get("criterion_used", "")).strip()}

def expected_answer(task):
    return str(task.expected_answer["answer"])

def score_answer(task, parsed):
    expected = expected_answer(task)
    if parsed is None:
        return {"predicted_answer": None, "criterion_used": None, "overall_correct": 0}
    predicted = parsed.get("answer")
    return {
        "predicted_answer": predicted,
        "criterion_used": parsed.get("criterion_used"),
        "overall_correct": int(predicted == expected),
    }

def random_baseline():
    return {"expected_accuracy": 0.5, "expected_pass_at_5": 1 - (0.5 ** 5), "expected_pass_5_all_correct": 0.5 ** 5}

def make_ground_truth_table(tasks):
    return pd.DataFrame([
        {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "pair_id": task.pair_id,
            "graph_A": task.graph_A,
            "graph_B": task.graph_B,
            "question": task.question,
            "expected_answer": expected_answer(task),
            "expected_details": json_dumps(task.expected_answer),
            "metadata": json_dumps(task.metadata),
        }
        for task in tasks
    ])

def summarize_by_paraphrase(results_df):
    rows = []
    group_cols = ["task_id", "task_type", "pair_id", "graph_A", "graph_B", "question", "expected_answer"]
    for keys, group in results_df.groupby(group_cols, dropna=False):
        task_id, task_type, pair_id, graph_A, graph_B, question, expected = keys
        correct_values = group["overall_correct"].astype(int).tolist()
        rows.append({
            "task_id": task_id,
            "task_type": task_type,
            "pair_id": pair_id,
            "graph_A": graph_A,
            "graph_B": graph_B,
            "question": question,
            "expected_answer": expected,
            "n_runs": len(group),
            "accuracy": sum(correct_values) / len(correct_values),
        })
    return pd.DataFrame(rows).sort_values(["task_type", "task_id"])

def summarize_by_task(results_df):
    rows = []
    group_cols = ["task_type", "pair_id", "graph_A", "graph_B", "expected_answer"]
    for keys, group in results_df.groupby(group_cols, dropna=False):
        task_type, pair_id, graph_A, graph_B, expected = keys
        correct_values = group["overall_correct"].astype(int).tolist()
        predicted_values = group["predicted_answer"].fillna("null").astype(str).tolist()
        rows.append({
            "task_type": task_type,
            "pair_id": pair_id,
            "graph_A": graph_A,
            "graph_B": graph_B,
            "expected_answer": expected,
            "n_trials": len(correct_values),
            "accuracy": sum(correct_values) / len(correct_values),
            "pass_at_5": int(any(v == 1 for v in correct_values)),
            "pass_5_all_correct": int(all(v == 1 for v in correct_values)),
            "consistency_answer": consistency_score(predicted_values),
            "consistency_overall_correctness": binary_consistency(correct_values),
        })
    return pd.DataFrame(rows).sort_values(["task_type"])

def summarize_by_task_type(task_summary_df):
  
    rows = []
    for task_type, group in task_summary_df.groupby("task_type", dropna=False):
        rows.append({
            "task_type": task_type,
            "n_tasks": int(len(group)),
            "accuracy": group["accuracy"].astype(float).mean(),
            "pass_at_5": group["pass_at_5"].astype(float).mean(),
            "pass_5_all_correct": group["pass_5_all_correct"].astype(float).mean(),
            "consistency_answer": group["consistency_answer"].astype(float).mean(),
            "consistency_overall_correctness": group["consistency_overall_correctness"].astype(float).mean(),
        })
    return pd.DataFrame(rows).sort_values(["task_type"])

def summarize_by_pair(results_df):
    rows = []
    group_cols = ["pair_id", "graph_A", "graph_B"]
    for keys, group in results_df.groupby(group_cols, dropna=False):
        pair_id, graph_A, graph_B = keys
        correct_values = group["overall_correct"].astype(int).tolist()
        rows.append({
            "pair_id": pair_id,
            "graph_A": graph_A,
            "graph_B": graph_B,
            "n_rows": len(group),
            "accuracy": sum(correct_values) / len(correct_values),
            "consistency_overall_correctness": binary_consistency(correct_values),
        })
    return pd.DataFrame(rows).sort_values(["pair_id"])

def answer_schema(task):
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["A", "B"]},
            "criterion_used": {"type": "string"},
        },
        "required": ["answer", "criterion_used"],
        "additionalProperties": False,
    }

def main():
    api_key = get_api_key()
    tasks = make_tasks()
    gt_df = make_ground_truth_table(tasks)
    gt_path = export_csv(gt_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="ground_truth")
    baseline_path = export_csv(pd.DataFrame([random_baseline()]), output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="random_baseline")
    print(f"Ground truth saved to: {gt_path}")
    print(gt_df[["task_id", "task_type", "pair_id", "graph_A", "graph_B", "question", "expected_answer"]].to_string(index=False))

    result_rows: List[Dict[str, Any]] = []
    total_calls = len(tasks) * N_RUNS
    call_idx = 0
    for task in tasks:
        prompt = build_prompt(task)
        for run_idx in range(1, N_RUNS + 1):
            call_idx += 1
            print(f"\n[{call_idx}/{total_calls}] {task.task_id} run {run_idx}/{N_RUNS}")
            raw_answer = call_claude(api_key=api_key, user_prompt=prompt, system_prompt=SYSTEM_PROMPT, config=CONFIG, answer_schema=answer_schema(task))
            parsed = parse(raw_answer)
            scored = score_answer(task, parsed)
            row = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "pair_id": task.pair_id,
                "graph_A": task.graph_A,
                "graph_B": task.graph_B,
                "run_idx": run_idx,
                "question": task.question,
                "expected_answer": expected_answer(task),
                "expected_details": json_dumps(task.expected_answer),
                "raw_answer": raw_answer,
                "parsed_answer": json_dumps(parsed) if parsed is not None else None,
                "predicted_answer": scored["predicted_answer"],
                "criterion_used": scored["criterion_used"],
                "overall_correct": scored["overall_correct"],
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "seed": SEED,
            }
            result_rows.append(row)
            export_csv(pd.DataFrame(result_rows), output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="results_partial")
            print("Raw answer:")
            print(raw_answer)
            print(f"Expected: {expected_answer(task)} | Predicted: {scored['predicted_answer']} | Correct: {scored['overall_correct']}")
            print(f"Criterion used: {scored['criterion_used']}")
            pause_between_runs(WAIT_SECONDS)

    results_df = pd.DataFrame(result_rows)
    results_path = export_csv(results_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="results")
    paraphrase_summary_df = summarize_by_paraphrase(results_df)
    paraphrase_summary_path = export_csv(paraphrase_summary_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="summary_by_paraphrase")
    task_summary_df = summarize_by_task(results_df)
    task_summary_path = export_csv(task_summary_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="summary_by_question")
    type_summary_df = summarize_by_task_type(task_summary_df)
    type_summary_path = export_csv(type_summary_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="summary_by_type")
    pair_summary_df = summarize_by_pair(results_df)
    pair_summary_path = export_csv(pair_summary_df, output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="summary_by_pair")
    raw_log_path = export_csv(results_df[["task_id", "task_type", "pair_id", "graph_A", "graph_B", "run_idx", "raw_answer", "parsed_answer", "expected_answer", "predicted_answer", "criterion_used", "overall_correct"]], output_dir=OUTPUT_DIR, module_name=MODULE_NAME, artifact_name="raw_answers")

    print("\nSaved outputs:")
    for path in [gt_path, baseline_path, results_path, paraphrase_summary_path, task_summary_path, type_summary_path, pair_summary_path, raw_log_path]:
        print(f"- {path}")
    print("\nSummary by paraphrase:")
    print(paraphrase_summary_df.to_string(index=False))
    print("\nSummary by task:")
    print(task_summary_df.to_string(index=False))
    print("\nSummary by task type:")
    print(type_summary_df.to_string(index=False))
    print("\nSummary by pair:")
    print(pair_summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
