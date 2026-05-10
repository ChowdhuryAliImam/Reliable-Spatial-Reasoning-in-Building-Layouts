from __future__ import annotations

"""
Semantic design-reasoning benchmark for Claude with:
- previous metric/path tools
- semantic KG query tools
- compact semantic rule set in the prompt
- tool_use loop until stop_reason == "end_turn"
- same parsing, scoring, and reporting structure as before

"""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

import connectivity_graph_data
import Tools

MODEL_NAME = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
N_RUNS = 1
WAIT_SECONDS = float(os.getenv("WAIT_SECONDS", "3"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "900"))
MODULE_NAME = "conceptual_kg"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", SCRIPT_DIR)
SEED = int(os.getenv("SEED", "42"))

graphs: Dict[str, Dict[str, List[str]]] = {
    "graph_1": connectivity_graph_data.graph_1,
    "graph_3": connectivity_graph_data.graph_3,
    "graph_4": connectivity_graph_data.graph_4,
}

# Task class

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

# Api key

def get_api_key():
    if os.getenv("SKIP_API_KEY_PROMPT") == "1":
        key = os.getenv("ANTHROPIC_API_KEY", "")
    else:
        print("Enter API key:")
        key = input().strip()

    key = key.strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("Missing API key.")

    os.environ["ANTHROPIC_API_KEY"] = key
    print(f"API key loaded. Length: {len(key)} characters")
    return key

# Semantic kg rules and queries
Triple = Tuple[str, str, str]

def infer_space_type(node):
    n = node.lower()

    if n.startswith("bed_room"):
        return "bedroom"
    if n.startswith("bath_room") or n.startswith("bathroom") or n.startswith("toilet"):
        return "bath_room"
    if n.startswith("sleeping_porch"):
        return "sleeping_porch"
    if n.startswith("living_room"):
        return "living_room"
    if n.startswith("dining_room"):
        return "dining_room"
    if n.startswith("kitchen"):
        return "kitchen"
    if n.startswith("hall"):
        return "hall"
    if n.startswith("sun_room") or n.startswith("sunroom"):
        return "sun_room"
    if "service_entry" in n:
        return "service_entry"
    if "entry" in n:
        return "entry"
    if "porch" in n:
        return "porch"
    if n.startswith("stair") or n.startswith("stairs"):
        return "stair"
    if n.startswith("landing"):
        return "landing"
    if n.startswith("pantry"):
        return "pantry"
    if n.startswith("vestibule"):
        return "vestibule"

    return "unknown"

def privacy_category(space_type):
    public = {"living_room", "porch", "sun_room", "entry", "vestibule"}
    semi_private = {"dining_room", "kitchen", "service_entry", "hall", "stair", "landing", "pantry"}
    private = {"bedroom", "sleeping_porch", "bath_room"}

    if space_type in public:
        return "public"
    if space_type in semi_private:
        return "semi_private"
    if space_type in private:
        return "private"
    return "neutral"

def access_role(space_type):
    if space_type in {"living_room", "dining_room", "hall"}:
        return "shared_or_guest_facing"
    if space_type in {"entry", "porch", "sun_room", "vestibule"}:
        return "public_access"
    if space_type in {"kitchen", "service_entry", "pantry"}:
        return "family_or_service"
    if space_type in {"bedroom", "bath_room", "sleeping_porch"}:
        return "restricted"
    if space_type in {"stair", "landing"}:
        return "circulation"
    return "unspecified"

def is_entry_type(space_type):
    return space_type in {"entry", "porch", "service_entry", "vestibule"}

def is_private_type(space_type):
    return privacy_category(space_type) == "private"

def is_shared_or_guest_facing_type(space_type):
    return access_role(space_type) == "shared_or_guest_facing"

def is_circulation_type(space_type):
    return space_type in {"hall", "stair", "landing"}

def build_semantic_kg_triples(graph, include_connectivity_edges = True):
    triples: List[Triple] = []

    for node in sorted(graph.keys()):
        space_type = infer_space_type(node)
        pcat = privacy_category(space_type)
        arole = access_role(space_type)

        triples.append((node, "has_type", space_type))
        triples.append((node, "has_privacy_category", pcat))
        triples.append((node, "has_access_role", arole))

        if is_entry_type(space_type):
            triples.append((node, "has_role", "entry"))
        if is_private_type(space_type):
            triples.append((node, "has_role", "private_space"))
        if is_shared_or_guest_facing_type(space_type):
            triples.append((node, "has_role", "shared_or_guest_facing_space"))
        if is_circulation_type(space_type):
            triples.append((node, "has_role", "circulation"))

    if include_connectivity_edges:
        seen_edges = set()
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                edge_key = tuple(sorted((node, neighbor)))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                triples.append((node, "connected_to", neighbor))
                triples.append((neighbor, "connected_to", node))

    return triples

def query_by_relation(triples, relation, tail):
    return sorted([head for head, rel, t in triples if rel == relation and t == tail])

def get_entries_from_kg(triples):
    return query_by_relation(triples, "has_role", "entry")

def get_private_spaces_from_kg(triples):
    return query_by_relation(triples, "has_role", "private_space")

def get_shared_or_guest_facing_spaces_from_kg(triples):
    return query_by_relation(triples, "has_role", "shared_or_guest_facing_space")

def get_circulation_spaces_from_kg(triples):
    return query_by_relation(triples, "has_role", "circulation")

SEMANTIC_RULE_SET = {
    "private_space_exposure": {
        "goal": "Protect private spaces from guest-facing or shared circulation.",
        "principle": "Compare entry-to-private shortest paths; fewer shared/guest-facing intermediate spaces is better.",
        "use": ["get_entries", "get_private_spaces", "get_shared_or_guest_facing_spaces", "calculate_shortest_path_info"],
    },
    "bottleneck_reduction": {
        "goal": "Avoid concentrating movement through a single space.",
        "principle": "Compare choice values; lower maximum choice means weaker bottlenecking.",
        "use": ["calculate_choice"],
    },
    "entry_access_efficiency": {
        "goal": "Make rooms efficiently reachable from valid entries.",
        "principle": "Compare shortest-path distances from entries to non-entry spaces; lower average distance is better.",
        "use": ["get_entries", "calculate_shortest_path_info"],
    },
}

TASK_TO_RULE_KEY = {
    "S1_private_space_exposure": "private_space_exposure",
    "S2_bottleneck_reduction": "bottleneck_reduction",
    "S3_entry_access_efficiency": "entry_access_efficiency",
}

# SEMANTIC CLASSIFICATION / KG-LIKE RULES FOR GROUND TRUTH

def is_entry(node):
    n = node.lower()
    return ("entry" in n) or ("porch" in n and "sleeping" not in n)

def is_private(node):
    n = node.lower()
    return (
        n.startswith("bed_room")
        or n.startswith("bath_room")
        or n.startswith("sleeping_porch")
    )

def is_shared_or_guest_facing(node):
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

# Metric / tool wrappers for ground truth

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

# Pair generation

def generate_pairs():
    return [
        ("graph_1", "graph_4"),  # S1
        ("graph_4", "graph_3"),  # S2
        ("graph_3", "graph_1"),  # S3
    ]

# Semantic paraphrases
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

# Ground truth

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

# Prompts

SYSTEM_PROMPT = """
You must use the provided tools when graph metrics, paths, or semantic categories are needed.
You must submit the final answer by calling the submit_answer tool.
Do not answer in free text.
Choose exactly one option: A or B.
""".strip()

def graph_to_json(adj):
    normalized = {node: sorted(set(neighbors)) for node, neighbors in sorted(adj.items())}
    return json.dumps(normalized, indent=2)

def build_prompt(task):
    rule_key = TASK_TO_RULE_KEY[task.task_type]
    semantic_rule = SEMANTIC_RULE_SET[rule_key]

    return f"""
Layout A ({task.graph_A}):
{graph_to_json(graphs[task.graph_A])}

Layout B ({task.graph_B}):
{graph_to_json(graphs[task.graph_B])}

Semantic rule:
{json.dumps(semantic_rule, indent=2)}

Question:
{task.question}

Use the available tools as needed. Then use the submit_answer tool with exactly these fields:
answer: "A" or "B"
criterion_used: a short phrase describing the criterion you used
""".strip()

# Api tool schemas and execution

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
            "name": "get_entries",
            "description": "Return valid entry spaces for Layout A or Layout B using the semantic KG.",
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
            "name": "get_private_spaces",
            "description": "Return private spaces for Layout A or Layout B using the semantic KG.",
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
            "name": "get_shared_or_guest_facing_spaces",
            "description": "Return shared or guest-facing spaces for Layout A or Layout B using the semantic KG.",
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
            "name": "get_circulation_spaces",
            "description": "Return circulation spaces for Layout A or Layout B using the semantic KG.",
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
            "name": "submit_answer",
            "description": "Submit the selected layout for the semantic design-comparison task.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "enum": ["A", "B"],
                        "description": "The selected layout.",
                    },
                    "criterion_used": {
                        "type": "string",
                        "description": "Short phrase describing the design criterion used.",
                    },
                },
                "required": ["answer", "criterion_used"],
                "additionalProperties": False,
            },
        },
    ]

def get_layout_graph(task, layout):
    if layout == "A":
        return graphs[task.graph_A]
    if layout == "B":
        return graphs[task.graph_B]
    raise ValueError(f"Unknown layout: {layout}")

def get_layout_triples(task, layout):
    return build_semantic_kg_triples(get_layout_graph(task, layout))

def execute_tool(tool_name, tool_input, task):
    layout = tool_input["layout"]
    adj = get_layout_graph(task, layout)

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

    triples = get_layout_triples(task, layout)

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

    messages = [{"role": "user", "content": user}]
    submitted_answer: Optional[Dict[str, Any]] = None
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            while True:
                payload = {
                    "model": MODEL_NAME,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "system": system,
                    "messages": messages,
                    "tools": make_tools(),
                }

                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

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
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": "Structured answer received.",
                            })
                            continue

                        try:
                            result = execute_tool(tool_name, tool_input, task)
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

                if stop_reason == "end_turn":
                    if submitted_answer is not None:
                        return json.dumps(submitted_answer, ensure_ascii=False)

                    texts = [
                        block.get("text", "")
                        for block in content_blocks
                        if block.get("type") == "text"
                    ]
                    return "\n".join(texts).strip()

                messages.append({
                    "role": "user",
                    "content": "Continue until complete. Use submit_answer for the final structured answer.",
                })

        except requests.HTTPError as exc:
            last_error = exc
            status = getattr(exc.response, "status_code", None)
            text = getattr(exc.response, "text", "")
            print(f"HTTP error on attempt {attempt}/{MAX_RETRIES}: {status} {text[:300]}")

            if status in {400, 401, 403, 404}:
                raise

            time.sleep(WAIT_SECONDS * (2 ** (attempt - 1)))

        except requests.RequestException as exc:
            last_error = exc
            print(f"Request error on attempt {attempt}/{MAX_RETRIES}: {exc}")
            time.sleep(WAIT_SECONDS * (2 ** (attempt - 1)))

    raise RuntimeError(f"Claude API call failed after {MAX_RETRIES} retries: {last_error}")

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

# Logging and summaries

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
            "consistency_overall_correctness": binary_consistency(correct_values),
            "consistency_answer": consistency_score(predicted_values),
            "random_baseline_accuracy": 0.5,
            "above_random_baseline": accuracy - 0.5,
        })

    return pd.DataFrame(rows).sort_values(["task_type"])

def summarize_by_task_type_from_question(question_summary_df):
    
    rows = []
    for task_type, group in question_summary_df.groupby("task_type", dropna=False):
        rows.append({
            "task_type": task_type,
            "n_questions": int(len(group)),
            "accuracy": group["accuracy_over_paraphrases"].astype(float).mean(),
            "accuracy_over_paraphrases": group["accuracy_over_paraphrases"].astype(float).mean(),
            "pass_at_5": group["pass_at_5_task"].astype(float).mean(),
            "pass_5_all_correct": group["pass_5_all_correct_task"].astype(float).mean(),
            "consistency_overall_correctness": group["consistency_overall_correctness"].astype(float).mean(),
            "consistency_answer": group["consistency_answer"].astype(float).mean(),
            "random_baseline_accuracy": group["random_baseline_accuracy"].astype(float).mean(),
            "above_random_baseline": group["above_random_baseline"].astype(float).mean(),
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

# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    api = get_api_key()
    tasks = make_tasks()

    gt_df = make_ground_truth_table(tasks)
    gt_path = os.path.join(OUTPUT_DIR, f"{MODULE_NAME}_ground_truth.csv")
    gt_df.to_csv(gt_path, index=False)

    print(f"Ground truth saved to: {gt_path}")
    print(gt_df[["task_id", "task_type", "pair_id", "graph_A", "graph_B", "question", "expected_answer"]].to_string(index=False))

    baseline_path = os.path.join(OUTPUT_DIR, f"{MODULE_NAME}_random_baseline.csv")
    pd.DataFrame([random_baseline()]).to_csv(baseline_path, index=False)

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
            partial_path = os.path.join(OUTPUT_DIR, f"{MODULE_NAME}_results_partial.csv")
            partial_df.to_csv(partial_path, index=False)

            print("Raw answer:")
            print(raw_answer)
            print(f"Expected: {expected_answer(task)} | Predicted: {scored['predicted_answer']} | Correct: {scored['overall_correct']}")
            print(f"Criterion used: {scored['criterion_used']}")

            time.sleep(WAIT_SECONDS)

    results_df = pd.DataFrame(result_rows)

    results_path = os.path.join(OUTPUT_DIR, f"{MODULE_NAME}_results.csv")
    results_df.to_csv(results_path, index=False)

    paraphrase_summary_df = summarize_by_paraphrase(results_df)
    paraphrase_summary_path = os.path.join(OUTPUT_DIR, f"{MODULE_NAME}_summary_by_paraphrase.csv")
    paraphrase_summary_df.to_csv(paraphrase_summary_path, index=False)
