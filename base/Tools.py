from __future__ import annotations

from typing import Dict, List
import networkx as nx


AdjacencyDict = Dict[str, List[str]]


def build_nx_graph(graph) :
   
    G = nx.Graph()

    for node, neighbors in graph.items():
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    return G


def calculate_mean_depth(graph) :
 
    G = build_nx_graph(graph)
    mean_depth = {}

    for node in G.nodes:
        lengths = nx.single_source_shortest_path_length(G, node)

        # Exclude distance to itself
        distances = [
            dist for target, dist in lengths.items()
            if target != node
        ]

        if len(distances) == 0:
            mean_depth[node] = 0.0
        else:
            mean_depth[node] = sum(distances) / len(distances)

    return dict(sorted(mean_depth.items()))

def calculate_choice(graph: AdjacencyDict, normalized: bool = False) -> Dict[str, float]:
  
    G = build_nx_graph(graph)

    choice = nx.betweenness_centrality(
        G,
        normalized=normalized,
        weight=None
    )

    return dict(sorted(choice.items()))

def calculate_shortest_path_info(graph, source, target):

    import networkx as nx

    G = build_nx_graph(graph)

    if source not in G:
        raise ValueError(f"Source node not found: {source}")
    if target not in G:
        raise ValueError(f"Target node not found: {target}")

    path = nx.shortest_path(G, source=source, target=target)

    return {
        "source": source,
        "target": target,
        "distance": len(path) - 1,
        "path": path,
    }

def get_neighbors(graph, node):
 
    G = build_nx_graph(graph)

    if node not in G:
        raise ValueError(f"Node not found: {node}")

    return sorted(list(G.neighbors(node)))

def get_degree(graph: AdjacencyDict, node: str) -> int:
 
    G = build_nx_graph(graph)

    if node not in G:
        raise ValueError(f"Node not found: {node}")

    return int(G.degree[node])


def get_nodes_within_k_steps(graph: AdjacencyDict,source: str,k: int = 2,include_source = False) :
 
    G = build_nx_graph(graph)

    if source not in G:
        raise ValueError(f"Source node not found: {source}")

    lengths = nx.single_source_shortest_path_length(G, source, cutoff=k)

    nodes = [
        node for node, distance in lengths.items()
        if distance <= k
    ]

    if not include_source:
        nodes = [node for node in nodes if node != source]

    return sorted(nodes)