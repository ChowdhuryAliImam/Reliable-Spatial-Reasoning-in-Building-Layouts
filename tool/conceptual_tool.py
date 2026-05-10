from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import connectivity_graph_data
import Tools
from tool_call_helper import call_with_tool_use, get_api_key as helper_get_api_key
from export_utils import ensure_output_dir, export_csv

MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
N_RUNS = 1  
WAIT_SECONDS = float(os.getenv("WAIT_SECONDS", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3000"))
MODULE_NAME = "conceptual_tool"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR_ENV = os.getenv("OUTPUT_DIR", "").strip()
OUTPUT_DIR = SCRIPT_DIR if not _OUTPUT_DIR_ENV or _OUTPUT_DIR_ENV == "." else _OUTPUT_DIR_ENV
SEED = int(os.getenv("SEED", "42"))

graphs: Dict[str, Dict[str, List[str]]] = {
    "graph_1": connectivity_graph_data.graph_1,
    "graph_3": connectivity_graph_data.graph_3,
    "graph_4": connectivity_graph_data.graph_4,
}

#Task class
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

# API key
def get_api_key():
    return helper_get_api_key("Enter API key:")

# Semantic Classification
def is_entry(node):
    """
    Valid entries:
    - any node containing 'entry'
    - any porch, except sleeping_porch
    """
    n = node.lower()
    return ("entry" in n) or ("porch" in n and "sleeping" not in n)

def is_private(node):
    """
    Private spaces for this benchmark:
    - bed_room*
    - bath_room*
    - sleeping_porch*
    """
    n = node.lower()
    return (
        n.startswith("bed_room")
        or n.startswith("bath_room")
        or n.startswith("sleeping_porch")
    )

def is_shared_or_guest_facing(node):
    """
    Shared / guest-facing spaces for S1 exposure calculation.

    Note: kitchen is intentionally excluded here to keep this benchmark
    context-neutral. If a project context treats kitchen as guest-facing,
    add `or n.startswith("kitchen")`.
    """
    n = node.lower()
    return (
        n.startswith("living_room")
        or n.startswith("dining_room")
        or n.startswith("hall")
    )

def valid_entries(adj):
    return sorted([node for node in adj if is_entry(node)])

def private_spaces(adj):
    return sorted([node for node in adj if is_private(node)])

def non_entry_spaces(adj):
    return sorted([node for node in adj if not is_entry(node)])

# Metric/ Tool Wrappers
def call_tool_choice(adj):
    if hasattr(Tools, "calculate_choice"):
        try:
            return Tools.calculate_choice(adj, normalized=False)
        except TypeError:
            return Tools.calculate_choice(adj)
    raise AttributeError("Tools.py must define calculate_choice(graph, normalized=False).")

def call_tool_shortest_path_info(adj, source, target):
    if hasattr(Tools, "calculate_shortest_path_info"):
        return Tools.calculate_shortest_path_info(adj, source, target)
    raise AttributeError("Tools.py must define calculate_shortest_path_info(graph, source, target).")

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

        nearest = min(
            call_tool_shortest_distance(adj, entry, target)
            for entry in candidate_entries
        )
        distances.append(nearest)

    return sum(distances) / len(distances)

def private_exposure_score(adj):
    """
    Lower is better:
    average number of shared/guest-facing spaces on shortest paths
    from valid entries to private spaces.
    """
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

        exposure_count = sum(
            1 for node in intermediate_nodes
            if is_shared_or_guest_facing(node)
        )

        exposure_counts.append(float(exposure_count))

    return sum(exposure_counts) / len(exposure_counts)

def bottleneck_score(adj):
    """
    Lower is better:
    weaker bottleneck effect measured as lower maximum unnormalized choice.
    """
    choice_values = call_tool_choice(adj)
    return max(float(v) for v in choice_values.values())

def entry_access_score(adj):
    """
    Lower is better:
    average nearest-entry distance to all non-entry spaces.
    """
    return average_nearest_entry_distance(adj, non_entry_spaces(adj))

def compare_scores(value_A, value_B, higher_is_better):
    if value_A == value_B:
        return "tie"
    if higher_is_better:
        return "A" if value_A > value_B else "B"
    return "A" if value_A < value_B else "B"

# Pair Generation
def generate_pairs():
    """
    Three graphs produce three unique pairs.
    Each semantic task uses one fixed pair.
    """
    return [
        ("graph_1", "graph_4"),  # S1
        ("graph_4", "graph_3"),  # S2
        ("graph_3", "graph_1"),  # S3
    ]

# Semantic Paraphrases
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

# Ground Truth

def compute_ground_truth_for_pair(task_type, graph_A, graph_B):
    adj_A = graphs[graph_A]
    adj_B = graphs[graph_B]

    if task_type == "S1_private_space_exposure":
        score_A = private_exposure_score(adj_A)
        score_B = private_exposure_score(adj_B)
        winner = compare_scores(score_A, score_B, higher_is_better=False)
        return {
            "answer": winner,
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
        winner = compare_scores(score_A, score_B, higher_is_better=False)
        return {
            "answer": winner,
            "criterion": "lower maximum unnormalized choice value",
            "layout_A_score": score_A,
            "layout_B_score": score_B,
            "higher_is_better": False,
        }

    if task_type == "S3_entry_access_efficiency":
        score_A = entry_access_score(adj_A)
        score_B = entry_access_score(adj_B)
        winner = compare_scores(score_A, score_B, higher_is_better=False)
        return {
            "answer": winner,
            "criterion": "lower average nearest-entry distance to all non-entry spaces",
            "layout_A_score": score_A,
            "layout_B_score": score_B,
            "higher_is_better": False,
            "layout_A_entries": valid_entries(adj_A),
            "layout_B_entries": valid_entries(adj_B),
        }

    raise ValueError(f"Unknown task type: {task_type}")

def make_tasks():
    """
    Create 15 task rows:
    - 3 semantic tasks
    - each with 5 paraphrases
    - each task uses one fixed pair
    """
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
            tasks.append(
                Task(
                    task_id=f"{task_short}_p{paraphrase_idx}",
                    task_type=task_type,
                    pair_id=f"{task_short}_pair",
                    graph_A=graph_A,
                    graph_B=graph_B,
                    question=question,
                    expected_answer=expected,
                    metadata={
                        "paraphrase_index": paraphrase_idx,
                        "seed": SEED,
                        "graph_A": graph_A,
                        "graph_B": graph_B,
                    },
                )
            )

    return tasks

# Promopts
SYSTEM_PROMPT = """
You must use the provided tools when graph metrics or paths are needed.
You must submit the final answer by calling the submit_answer tool.
Do not answer in free text.
Choose exactly one option: A or B.
""".strip()

def graph_to_json(adj):
    normalized = {node: sorted(set(neighbors)) for node, neighbors in sorted(adj.items())}
    return json.dumps(normalized, indent=2)

def semantic_context_for_prompt(task):
    base_context = """
Semantic definitions for this benchmark:
- Valid entries are any node containing "entry" or any porch, except sleeping_porch.
- Private spaces are bed_room*, bath_room*, and sleeping_porch*.
- Shared or guest-facing spaces are living_room*, dining_room*, and hall*.
""".strip()

    if task.task_type == "S1_private_space_exposure":
        return base_context + """

Task interpretation:
- Better private-space protection means private spaces are reached with fewer passes through shared or guest-facing spaces.
""".strip()

    if task.task_type == "S2_bottleneck_reduction":
        return base_context + """

Task interpretation:
- A bottleneck means movement is overly concentrated through one space.
- The better layout is the one with weaker bottleneck effect.
""".strip()

    if task.task_type == "S3_entry_access_efficiency":
        return base_context + """

Task interpretation:
- Better entry access means rooms are more efficiently reachable from valid entry spaces.
""".strip()

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

Use the submit_answer tool with exactly these fields:
answer: "A" or "B"
criterion_used: a short phrase describing the criterion you used
""".strip()

# Api call

def make_tools():
    return [
        {
            "name": "calculate_mean_depth",
            "description": "Calculate mean depth for every node in Layout A or Layout B.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "layout": {"type": "string", "enum": ["A", "B"]},
                },
                "required": ["layout"],
                "additionalProperties": False,
            },
        },
        {
            "name": "calculate_choice",
            "description": "Calculate choice/betweenness centrality for every node in Layout A or Layout B.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "layout": {"type": "string", "enum": ["A", "B"]},
                    "normalized": {"type": "boolean", "default": False},
                },
                "required": ["layout"],
                "additionalProperties": False,
            },
        },
        {
            "name": "calculate_shortest_path_info",
            "description": "Return shortest path and distance between two nodes in Layout A or Layout B.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "layout": {"type": "string", "enum": ["A", "B"]},
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["layout", "source", "target"],
                "additionalProperties": False,
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the selected layout for the semantic design-comparison task.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "enum": ["A", "B"],
                    },
                    "criterion_used": {
                        "type": "string",
                    },
                },
                "required": ["answer", "criterion_used"],
                "additionalProperties": False,
            },
        },
    ]

def execute_tool(tool_name, tool_input, task):
    layout = tool_input["layout"]

    if layout == "A":
        adj = graphs[task.graph_A]
    elif layout == "B":
        adj = graphs[task.graph_B]
    else:
        raise ValueError(f"Unknown layout: {layout}")

    if tool_name == "calculate_mean_depth":
        return Tools.calculate_mean_depth(adj)

    if tool_name == "calculate_choice":
        return Tools.calculate_choice(
            adj,
            normalized=bool(tool_input.get("normalized", False)),
        )

    if tool_name == "calculate_shortest_path_info":
        return Tools.calculate_shortest_path_info(
            adj,
            tool_input["source"],
            tool_input["target"],
        )

    raise ValueError(f"Unknown executable tool: {tool_name}")

def call(api_key, system, user, task):
    return call_with_tool_use(
        api_key=api_key,
        user_prompt=user,
        system_prompt=system,
        tools=make_tools(),
        execute_tool=lambda tool_name, tool_input: execute_tool(tool_name, tool_input, task),
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
def parse(raw_text):
    if raw_text is None or not str(raw_text).strip():
        return None

    try:
        parsed_json = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed_json, dict):
        return None

    answer = parsed_json.get("answer")
    if isinstance(answer, str):
        answer = answer.strip().upper()

    if answer not in {"A", "B"}:
        return None

    return {
        "answer": answer,
        "criterion_used": str(parsed_json.get("criterion_used", "")).strip(),
    }

# Scoring
def expected_answer(task):
    return str(task.expected_answer["answer"])

def score_answer(task, parsed):
    expected = expected_answer(task)

    if parsed is None:
        return {
            "predicted_answer": None,
            "criterion_used": None,
            "overall_correct": 0,
        }

    predicted = parsed.get("answer")
    correct = int(predicted == expected)

    return {
        "predicted_answer": predicted,
        "criterion_used": parsed.get("criterion_used"),
        "overall_correct": correct,
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

def random_baseline():
    return {
        "expected_accuracy": 0.5,
        "expected_pass_at_5": 1 - (0.5 ** 5),
        "expected_pass_5_all_correct": 0.5 ** 5,
    }

def make_ground_truth_table(tasks):
    rows = []
    for task in tasks:
        rows.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "pair_id": task.pair_id,
            "graph_A": task.graph_A,
            "graph_B": task.graph_B,
            "question": task.question,
            "expected_answer": expected_answer(task),
            "expected_details": json.dumps(task.expected_answer, ensure_ascii=False),
            "metadata": json.dumps(task.metadata, ensure_ascii=False),
        })
    return pd.DataFrame(rows)

def summarize_by_paraphrase(results_df):
    """
    Per-paraphrase logging. This is NOT the main PASS@5 summary.
    Each row has one run because each paraphrase is a repeated trial.
    """
    rows = []
    group_cols = [
        "task_id",
        "task_type",
        "pair_id",
        "graph_A",
        "graph_B",
        "question",
        "expected_answer",
    ]

    for keys, group in results_df.groupby(group_cols, dropna=False):
        task_id, task_type, pair_id, graph_A, graph_B, question, expected = keys
        correct_values = group["overall_correct"].astype(int).tolist()
        accuracy = sum(correct_values) / len(correct_values)

        rows.append({
            "task_id": task_id,
            "task_type": task_type,
            "pair_id": pair_id,
            "graph_A": graph_A,
            "graph_B": graph_B,
            "question": question,
            "expected_answer": expected,
            "n_runs": len(group),
            "accuracy": accuracy,
        })

    return pd.DataFrame(rows).sort_values(["task_type", "task_id"])

def summarize_by_task(results_df):
    """
    Main semantic-task summary.

    Paraphrases are treated as repeated trials of the SAME task.
    PASS@5 and PASS^5 are computed across the five paraphrases.
    """
    rows = []
    group_cols = [
        "task_type",
        "pair_id",
        "graph_A",
        "graph_B",
        "expected_answer",
    ]

    for keys, group in results_df.groupby(group_cols, dropna=False):
        task_type, pair_id, graph_A, graph_B, expected = keys
        correct_values = group["overall_correct"].astype(int).tolist()
        predicted_values = group["predicted_answer"].fillna("null").astype(str).tolist()

        accuracy = sum(correct_values) / len(correct_values)

        rows.append({
            "task_type": task_type,
            "pair_id": pair_id,
            "graph_A": graph_A,
            "graph_B": graph_B,
            "expected_answer": expected,
            "n_paraphrases": len(group),
            "accuracy_over_paraphrases": accuracy,
            "pass_at_5_task": pass_at_k(correct_values),
            "pass_5_all_correct_task": pass_all(correct_values),
            "pass_at_5": pass_at_k(correct_values),
            "pass_5_all_correct": pass_all(correct_values),
            "consistency_overall_correctness": binary_consistency(correct_values),
            "consistency_answer": consistency_score(predicted_values),
            "random_baseline_accuracy": 0.5,
            "above_random_baseline": accuracy - 0.5,
        })

    return pd.DataFrame(rows).sort_values(["task_type"])

def summarize_by_task_type(results_df):

    return summarize_by_task(results_df).copy()

def summarize_by_pair(results_df):
    rows = []
    for (pair_id, graph_A, graph_B), group in results_df.groupby(["pair_id", "graph_A", "graph_B"]):
        rows.append({
            "pair_id": pair_id,
            "graph_A": graph_A,
            "graph_B": graph_B,
            "n_rows": len(group),
            "accuracy": group["overall_correct"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["pair_id"])

#Main

def main():
    ensure_output_dir(OUTPUT_DIR)

    api = get_api_key()
    tasks = make_tasks()

    gt_df = make_ground_truth_table(tasks)
    gt_path = export_csv(gt_df, OUTPUT_DIR, MODULE_NAME, "ground_truth")

    print(f"Ground truth saved to: {gt_path}")
    print(gt_df[["task_id", "task_type", "pair_id", "graph_A", "graph_B", "question", "expected_answer"]].to_string(index=False))

    baseline_path = export_csv(pd.DataFrame([random_baseline()]), OUTPUT_DIR, MODULE_NAME, "random_baseline")

    result_rows: List[Dict[str, Any]] = []
    total_calls = len(tasks) * N_RUNS
    call_idx = 0

    for task in tasks:
        prompt = build_prompt(task)

        for run_idx in range(1, N_RUNS + 1):
            call_idx += 1
            print(f"\n[{call_idx}/{total_calls}] {task.task_id} run {run_idx}/{N_RUNS}")

            raw_answer = call(api, SYSTEM_PROMPT, prompt, task)
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
                "expected_details": json.dumps(task.expected_answer, ensure_ascii=False),
                "raw_answer": raw_answer,
                "parsed_answer": json.dumps(parsed, ensure_ascii=False) if parsed is not None else None,
                "predicted_answer": scored["predicted_answer"],
                "criterion_used": scored["criterion_used"],
                "overall_correct": scored["overall_correct"],
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "seed": SEED,
            }
            result_rows.append(row)

            partial_df = pd.DataFrame(result_rows)
            partial_path = export_csv(partial_df, OUTPUT_DIR, MODULE_NAME, "results_partial")

            print("Raw answer:")
            print(raw_answer)
            print(f"Expected: {expected_answer(task)} | Predicted: {scored['predicted_answer']} | Correct: {scored['overall_correct']}")
            print(f"Criterion used: {scored['criterion_used']}")

            time.sleep(WAIT_SECONDS)

    results_df = pd.DataFrame(result_rows)

    results_path = export_csv(results_df, OUTPUT_DIR, MODULE_NAME, "results")

    paraphrase_summary_df = summarize_by_paraphrase(results_df)
    paraphrase_summary_path = export_csv(paraphrase_summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_paraphrase")

    task_summary_df = summarize_by_task(results_df)
    task_summary_path = export_csv(task_summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_question")

    type_summary_df = summarize_by_task_type(results_df)
    type_summary_path = export_csv(type_summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_type")

    pair_summary_df = summarize_by_pair(results_df)
    pair_summary_path = export_csv(pair_summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_pair")

    raw_log_path = export_csv(
        results_df[[
            "task_id", "task_type", "pair_id", "graph_A", "graph_B", "run_idx",
            "raw_answer", "parsed_answer", "expected_answer", "predicted_answer",
            "criterion_used", "overall_correct"
        ]],
        OUTPUT_DIR,
        MODULE_NAME,
        "raw_answers",
    )

    print("\nSaved outputs:")
    print(f"- {gt_path}")
    print(f"- {baseline_path}")
    print(f"- {results_path}")
    print(f"- {paraphrase_summary_path}")
    print(f"- {task_summary_path}")
    print(f"- {type_summary_path}")
    print(f"- {pair_summary_path}")
    print(f"- {raw_log_path}")

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
