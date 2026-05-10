from __future__ import annotations

"""Local graph-reasoning benchmark for Claude using graph tools + semantic KG tools.

Tasks:
- L1 direct adjacency
- L2 degree comparison
- L3 two-step reachability

This is an experiment script, not only an export script. It calls Claude with tool_use,
allows local graph tools and semantic KG query tools, and forces final structured output
through the submit_answer tool.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
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

MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
N_RUNS = int(os.getenv("N_RUNS", "5"))
WAIT_SECONDS = float(os.getenv("WAIT_SECONDS", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
API_URL = os.getenv("ANTHROPIC_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3000"))
MODULE_NAME = "local_kg"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", SCRIPT_DIR)

graphs: Dict[str, Dict[str, List[str]]] = {
    "graph_1": connectivity_graph_data.graph_1,
    "graph_2": connectivity_graph_data.graph_2,
    "graph_3": connectivity_graph_data.graph_3,
    "graph_4": connectivity_graph_data.graph_4,
    "graph_5": connectivity_graph_data.graph_5,
}

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

def build_nx_graph(adj_list):
    graph = nx.Graph()
    for node, neighbors in adj_list.items():
        graph.add_node(node)
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)
    return graph

def neighbors_of(graph, node):
    return sorted_unique(list(graph.neighbors(node)))

def degree_winner(graph, node_a, node_b):
    degree_a = int(graph.degree[node_a])
    degree_b = int(graph.degree[node_b])
    if degree_a > degree_b:
        winner = node_a
    elif degree_b > degree_a:
        winner = node_b
    else:
        winner = "tie"
    return {"winner": winner, "degree_a": degree_a, "degree_b": degree_b}

def nodes_within_two_steps(graph, source, include_source = False):
    lengths = nx.single_source_shortest_path_length(graph, source, cutoff=2)
    nodes = [node for node, distance in lengths.items() if distance <= 2]
    if not include_source:
        nodes = [node for node in nodes if node != source]
    return sorted_unique(nodes)

def make_tasks():
    tasks: List[Task] = []
    for graph_name, adjacency in graphs.items():
        nx_graph = build_nx_graph(adjacency)

        room_x = L1_TARGETS[graph_name]
        tasks.append(Task(
            task_id=f"{graph_name}_L1",
            task_type="L1_direct_adjacency",
            graph_name=graph_name,
            question=f"Which spaces are directly connected to {room_x}?",
            expected_answer=neighbors_of(nx_graph, room_x),
            answer_format='{"answer": ["space_1", "space_2"]}',
            metadata={"room_x": room_x},
        ))

        room_a, room_b = L2_PAIRS[graph_name]
        tasks.append(Task(
            task_id=f"{graph_name}_L2",
            task_type="L2_degree_comparison",
            graph_name=graph_name,
            question=f"Which space has more direct connections: {room_a} or {room_b}?",
            expected_answer=degree_winner(nx_graph, room_a, room_b),
            answer_format='{"answer": "space_name_or_tie"}',
            metadata={"room_a": room_a, "room_b": room_b},
        ))

        room_x = L3_TARGETS[graph_name]
        tasks.append(Task(
            task_id=f"{graph_name}_L3",
            task_type="L3_two_step_reachability",
            graph_name=graph_name,
            question=f"Which spaces can be reached from {room_x} in two steps or fewer? Exclude {room_x} itself.",
            expected_answer=nodes_within_two_steps(nx_graph, room_x, include_source=False),
            answer_format='{"answer": ["space_1", "space_2"]}',
            metadata={"room_x": room_x, "include_source": False},
        ))
    return tasks

# Prompting

def graph_to_json(adjacency):
    normalized = {node: sorted_unique(neighbors) for node, neighbors in sorted(adjacency.items())}
    return json.dumps(normalized, indent=2)

def kg_triples_to_json(triples):
    rows = [{"head": h, "relation": r, "tail": t} for h, r, t in triples]
    return json.dumps(rows, indent=2, ensure_ascii=False)

def semantic_rule_set(task):
    if task.task_type == "L1_direct_adjacency":
        return {
            "goal": "Find spaces directly connected to the target room.",
            "use": ["get_neighbors", "get_node_profile", "submit_answer"],
            "answer_rule": "Return all direct neighbors sorted alphabetically.",
        }
    if task.task_type == "L2_degree_comparison":
        return {
            "goal": "Compare direct connectivity of two rooms.",
            "use": ["get_degree", "get_node_profile", "submit_answer"],
            "answer_rule": "Return the node with larger degree, or tie if equal.",
        }
    if task.task_type == "L3_two_step_reachability":
        return {
            "goal": "Find rooms reachable from the source within two steps.",
            "use": ["get_nodes_within_k_steps", "get_node_profile", "submit_answer"],
            "answer_rule": "Return all reachable rooms within k=2, excluding the source, sorted alphabetically.",
        }
    raise ValueError(f"Unknown task type: {task.task_type}")

SYSTEM_PROMPT = """
You are a graph-reasoning benchmark participant.
You must use the provided graph tools and semantic KG tools when useful.
You must submit the final answer by calling the submit_answer tool.
Do not answer in free text.
Use exact node names.
""".strip()

def build_prompt(task):
    adjacency = graphs[task.graph_name]
    triples = build_semantic_kg_triples(adjacency)
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
    <instruction>Use exact node names from the graph or KG.</instruction>
    <instruction>For list answers, include all correct spaces and sort them alphabetically.</instruction>
    <instruction>Use submit_answer for the final answer.</instruction>
  </instructions>
  <required_output_format>{task.answer_format}</required_output_format>
</benchmark_task>
""".strip()

# Tool schemas and execution

def make_tools(task):
    answer_schema: Dict[str, Any]
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
                "additionalProperties": False,
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
                "additionalProperties": False,
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
            "description": "Submit the final structured answer.",
            "input_schema": {
                "type": "object",
                "properties": {"answer": answer_schema},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    ]

def execute_tool(tool_name, tool_input):
    graph_name = tool_input["graph_name"]
    adjacency = graphs[graph_name]
    triples = build_semantic_kg_triples(adjacency)

    if tool_name == "get_neighbors":
        return Tools.get_neighbors(adjacency, tool_input["node"])

    if tool_name == "get_degree":
        return Tools.get_degree(adjacency, tool_input["node"])

    if tool_name == "get_nodes_within_k_steps":
        return Tools.get_nodes_within_k_steps(
            graph=adjacency,
            source=tool_input["source"],
            k=int(tool_input.get("k", 2)),
            include_source=bool(tool_input.get("include_source", False)),
        )

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
        expected = task.expected_answer["winner"]
        return int(model_answer == expected), model_answer

    raise ValueError(f"Unknown task type: {task.task_type}")

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
            "task_family": "local",
            "task_id": task.task_id,
            "graph_name": task.graph_name,
            "task_type": task.task_type,
            "question": task.question,
            "expected_answer": json.dumps(task.expected_answer, ensure_ascii=False),
            "metadata": json.dumps(task.metadata, ensure_ascii=False),
        }
        for task in tasks
    ])

def summarize_results(results_df):
    rows = []
    group_cols = ["condition", "task_family", "graph_name", "task_id", "task_type", "question", "expected_answer"]
    for keys, group in results_df.groupby(group_cols, dropna=False):
        condition, task_family, graph_name, task_id, task_type, question, expected_answer = keys
        correct_values = group["correct"].astype(int).tolist()
        rows.append({
            "condition": condition,
            "task_family": task_family,
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
            "condition": "kg",
            "task_family": "local",
            "task_type": task_type,
            "n_questions": int(len(group)),
            "accuracy": group["accuracy"].astype(float).mean(),
            "pass_at_5": group["pass_at_5"].astype(float).mean(),
            "pass_5_all_correct": group["pass_5_all_correct"].astype(float).mean(),
            "consistency_majority_correctness": group["consistency_majority_correctness"].astype(float).mean(),
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
    print(gt_df[["task_id", "question", "expected_answer"]].to_string(index=False))

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
            correct, normalized_model_answer = score_answer(task, parsed)

            row = {
                "condition": "kg",
                "task_family": "local",
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
            export_csv(pd.DataFrame(result_rows), OUTPUT_DIR, MODULE_NAME, "results_partial")
            print(f"Correct: {correct} | Answer: {normalized_model_answer}")
            time.sleep(WAIT_SECONDS)

    results_df = pd.DataFrame(result_rows)
    results_path = export_csv(results_df, OUTPUT_DIR, MODULE_NAME, "results")
    summary_df = summarize_results(results_df)
    summary_path = export_csv(summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_question")
    type_summary_df = summarize_by_task_type(summary_df)
    type_summary_path = export_csv(type_summary_df, OUTPUT_DIR, MODULE_NAME, "summary_by_type")
    raw_log_path = export_csv(
        results_df[["condition", "task_family", "task_id", "task_type", "graph_name", "run_idx", "raw_answer", "parsed_answer", "expected_answer", "normalized_model_answer", "correct"]],
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
