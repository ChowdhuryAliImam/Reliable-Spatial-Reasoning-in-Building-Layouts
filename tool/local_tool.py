

from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import pandas as pd
import connectivity_graph_data
from Tools import get_neighbors, get_degree, get_nodes_within_k_steps
from tool_call_helper import call_with_tool_use, get_api_key as helper_get_api_key
from export_utils import ensure_output_dir, export_csv

# Configuration

MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
N_RUNS = int(os.getenv("N_RUNS", "5"))
WAIT_SECONDS = float(os.getenv("WAIT_SECONDS", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
MODULE_NAME = "local_tool"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", SCRIPT_DIR)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3000"))
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Graph loading
graphs: Dict[str, Dict[str, List[str]]] = {
    "graph_1": connectivity_graph_data.graph_1,
    "graph_2": connectivity_graph_data.graph_2,
    "graph_3": connectivity_graph_data.graph_3,
    "graph_4": connectivity_graph_data.graph_4,
    "graph_5": connectivity_graph_data.graph_5,
}
# Task specification

L1_TARGETS = {
    "graph_1": "hall",
    "graph_2": "hall",
    "graph_3": "hall",
    "graph_4": "hall",
    "graph_5": "hall_1",
}

L2_PAIRS = {
    "graph_1": ("living_room", "dining_room"),
    "graph_2": ("living_room", "dining_room"),
    "graph_3": ("living_room", "dining_room"),
    "graph_4": ("living_room", "dining_room"),
    "graph_5": ("living_room_1", "dining_room_1"),
}

L3_TARGETS = {
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
    expected_answer: Any
    answer_format: str
    metadata: Dict[str, Any]

# Graph utilitie

def build_nx_graph(adj_list):
    G = nx.Graph()
    for node, neighbors in adj_list.items():
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    return G

def sorted_unique(items):
    return sorted(set(items))

def neighbors_of(G, node):
    return sorted_unique(list(G.neighbors(node)))

def degree_winner(G, a, b):
    deg_a = int(G.degree[a])
    deg_b = int(G.degree[b])
    if deg_a > deg_b:
        winner = a
    elif deg_b > deg_a:
        winner = b
    else:
        winner = "tie"
    return {"winner": winner, "degree_a": deg_a, "degree_b": deg_b}

def nodes_within_two_steps(G, source, include_source = False):
    lengths = nx.single_source_shortest_path_length(G, source, cutoff=2)
    nodes = [node for node, dist in lengths.items() if dist <= 2]
    if not include_source:
        nodes = [node for node in nodes if node != source]
    return sorted_unique(nodes)

def make_tasks():
    tasks: List[Task] = []
    for graph_name, adj in graphs.items():
        G = build_nx_graph(adj)

        x = L1_TARGETS[graph_name]
        tasks.append(Task(
            task_id=f"{graph_name}_L1",
            task_type="L1_direct_adjacency",
            graph_name=graph_name,
            question=f"Which spaces are directly connected to {x}?",
            expected_answer=neighbors_of(G, x),
            answer_format='{"answer": ["space_1", "space_2"]}',
            metadata={"room_x": x},
        ))

        a, b = L2_PAIRS[graph_name]
        tasks.append(Task(
            task_id=f"{graph_name}_L2",
            task_type="L2_degree_comparison",
            graph_name=graph_name,
            question=f"Which space has more direct connections: {a} or {b}?",
            expected_answer=degree_winner(G, a, b),
            answer_format='{"answer": "space_name_or_tie"}',
            metadata={"room_a": a, "room_b": b},
        ))

        x = L3_TARGETS[graph_name]
        tasks.append(Task(
            task_id=f"{graph_name}_L3",
            task_type="L3_two_step_reachability",
            graph_name=graph_name,
            question=f"Which spaces can be reached from {x} in two steps or fewer? Exclude {x} itself.",
            expected_answer=nodes_within_two_steps(G, x, include_source=False),
            answer_format='{"answer": ["space_1", "space_2"]}',
            metadata={"room_x": x, "include_source": False},
        ))
    return tasks

# Prompting

def build_prompt(task):
    return f"""
<benchmark_task>
  <role>You are evaluating a room connectivity graph.</role>

  <graph_name>{task.graph_name}</graph_name>

  <question>
{task.question}
  </question>

  <instructions>
    <instruction>You must use the provided tools to answer.</instruction>
    <instruction>Use graph_name exactly as provided.</instruction>
    <instruction>Use exact node names from the question.</instruction>
    <instruction>After using tools, call submit_answer with the final structured answer.</instruction>
    <instruction>Do not include markdown, prose, or explanation.</instruction>
    <instruction>For list answers, include all correct spaces and sort them alphabetically.</instruction>
    <instruction>For degree comparison, return the name of the space with more direct connections, or 'tie' if they are equal.</instruction>
  </instructions>

  <required_output_format>
{task.answer_format}
  </required_output_format>
</benchmark_task>
""".strip()

# Claude tool schemas

def make_tools(task):
    """
    Build tool schemas per task so submit_answer has an unambiguous concrete
    type for the answer field -- no oneOf, which is not reliably supported by
    the Anthropic API and causes the model to return free text instead of
    calling submit_answer (resulting in None answers from the grader).
    """
    if task.task_type in {"L1_direct_adjacency", "L3_two_step_reachability"}:
        answer_schema = {"type": "array", "items": {"type": "string"}}
    else:
        answer_schema = {"type": "string"}

    return [
        {
            "name": "get_neighbors",
            "description": "Return spaces directly connected to a node in a named graph.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string", "enum": list(graphs.keys())},
                    "node": {"type": "string"},
                },
                "required": ["graph_name", "node"],
            },
        },
        {
            "name": "get_degree",
            "description": "Return the number of direct connections for a node in a named graph.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string", "enum": list(graphs.keys())},
                    "node": {"type": "string"},
                },
                "required": ["graph_name", "node"],
            },
        },
        {
            "name": "get_nodes_within_k_steps",
            "description": "Return all spaces reachable from a source node within k steps.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_name": {"type": "string", "enum": list(graphs.keys())},
                    "source": {"type": "string"},
                    "k": {"type": "integer", "default": 2},
                    "include_source": {"type": "boolean", "default": False},
                },
                "required": ["graph_name", "source"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final structured answer for the local graph task.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": answer_schema,
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    ]

def execute_tool(tool_name, tool_input):
    graph_name = tool_input["graph_name"]
    graph = graphs[graph_name]

    if tool_name == "get_neighbors":
        return get_neighbors(graph, tool_input["node"])

    if tool_name == "get_degree":
        return get_degree(graph, tool_input["node"])

    if tool_name == "get_nodes_within_k_steps":
        return get_nodes_within_k_steps(
            graph=graph,
            source=tool_input["source"],
            k=int(tool_input.get("k", 2)),
            include_source=bool(tool_input.get("include_source", False)),
        )

    raise ValueError(f"Unknown tool: {tool_name}")

# Claude API via direct requests

def get_api_key():
    return helper_get_api_key("Enter your Anthropic API key:")

def build_system_prompt():
    return (
        "You are a graph-reasoning benchmark participant. "
        "You must use the provided tools to answer graph questions. "
        "Use exact node names. Return JSON only."
    )

def call_claude(api_key, user_prompt, task, system_prompt = ""):
    return call_with_tool_use(
        api_key=api_key,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
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
        continue_message="Continue until complete. You must call submit_answer with the final structured answer.",
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

def consistency_score(correct_values):
    if not correct_values:
        return 0.0
    ones = sum(correct_values)
    zeros = len(correct_values) - ones
    return max(ones, zeros) / len(correct_values)

# Logging and evaluation

def make_ground_truth_table(tasks):
    rows = []
    for task in tasks:
        rows.append({
            "task_id": task.task_id,
            "graph_name": task.graph_name,
            "task_type": task.task_type,
            "question": task.question,
            "expected_answer": json.dumps(task.expected_answer, ensure_ascii=False),
            "metadata": json.dumps(task.metadata, ensure_ascii=False),
        })
    return pd.DataFrame(rows)

def summarize_results(results_df):
    rows = []
    group_cols = ["graph_name", "task_id", "task_type", "question", "expected_answer"]

    for keys, group in results_df.groupby(group_cols, dropna=False):
        graph_name, task_id, task_type, question, expected_answer = keys
        correct_values = group["correct"].astype(int).tolist()
        accuracy = sum(correct_values) / len(correct_values)
        pass_at_5 = int(any(v == 1 for v in correct_values))
        pass_5 = int(all(v == 1 for v in correct_values))
        consistency = consistency_score(correct_values)

        rows.append({
            "graph_name": graph_name,
            "task_id": task_id,
            "task_type": task_type,
            "question": question,
            "expected_answer": expected_answer,
            "n_runs": len(correct_values),
            "accuracy": accuracy,
            "pass_at_5": pass_at_5,
            "pass_5_all_correct": pass_5,
            "consistency_majority_correctness": consistency,
        })

    return pd.DataFrame(rows).sort_values(["graph_name", "task_id"])

def summarize_by_task_type(question_summary_df):
    """Aggregate only from per-question/task summary rows."""
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

def main():
    ensure_output_dir(OUTPUT_DIR)

    tasks = make_tasks()
    gt_df = make_ground_truth_table(tasks)
    gt_path = export_csv(gt_df, OUTPUT_DIR, MODULE_NAME, "ground_truth")
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

            raw_answer = call_claude(api_key, prompt, task, system_prompt=system_prompt)
            parsed = extract_json(raw_answer)
            correct, normalized_model_answer = score_answer(task, parsed)

            row = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "graph_name": task.graph_name,
                "run_idx": run_idx,
                "question": task.question,
                "expected_answer": json.dumps(task.expected_answer, ensure_ascii=False),
                "raw_answer": raw_answer,
                "parsed_answer": json.dumps(parsed, ensure_ascii=False) if parsed is not None else None,
                "normalized_model_answer": json.dumps(normalized_model_answer, ensure_ascii=False),
                "correct": correct,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
            }
            result_rows.append(row)

            partial_df = pd.DataFrame(result_rows)
            partial_path = export_csv(partial_df, OUTPUT_DIR, MODULE_NAME, "results_partial")

            print(f"Correct: {correct} | Answer: {normalized_model_answer}")
            time.sleep(WAIT_SECONDS)

    results_df = pd.DataFrame(result_rows)
    results_path = export_csv(results_df, OUTPUT_DIR, MODULE_NAME, "results")

    summary_df = summarize_results(results_df)
    summary_path = export_csv(summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_question")

    type_summary_df = summarize_by_task_type(summary_df)
    type_summary_path = export_csv(type_summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_type")

    raw_log_df = results_df[[
        "task_id", "task_type", "graph_name", "run_idx", "raw_answer",
        "parsed_answer", "expected_answer", "normalized_model_answer", "correct",
        "model", "temperature",
    ]].copy()
    raw_log_path = export_csv(raw_log_df, OUTPUT_DIR, MODULE_NAME, "raw_answers")

    saved_paths = [gt_path, results_path, summary_path, type_summary_path, raw_log_path]

    print("\nSaved outputs:")
    for path in saved_paths:
        print(f"- {path}")

    print("\nSummary by question:")
    print(summary_df.to_string(index=False))

    print("\nSummary by task type:")
    print(type_summary_df.to_string(index=False))

if __name__ == "__main__":
    main()