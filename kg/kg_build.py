from __future__ import annotations

from typing import Dict, List, Tuple

Triple = Tuple[str, str, str]

def infer_space_type(node):
    n = node.lower()

    if n.startswith("bed_room") or n.startswith("new_bedroom"):
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

def tails_for_head(triples, head, relation):
    return sorted([tail for h, rel, tail in triples if h == head and rel == relation])

def get_entries_from_kg(triples):
    return query_by_relation(triples, "has_role", "entry")

def get_private_spaces_from_kg(triples):
    return query_by_relation(triples, "has_role", "private_space")

def get_shared_or_guest_facing_spaces_from_kg(triples):
    return query_by_relation(triples, "has_role", "shared_or_guest_facing_space")

def get_circulation_spaces_from_kg(triples):
    return query_by_relation(triples, "has_role", "circulation")
