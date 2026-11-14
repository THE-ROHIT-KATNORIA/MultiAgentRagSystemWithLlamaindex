"""
Node Cache — JSON
=================

Serialise / deserialise LlamaIndex TextNodes to JSON.
Replaces the old pickle-based cache (safer, version-stable).
"""

import os
import json
import logging
from config.settings import NODE_CACHE_DIR, NODE_CACHE_FILE

log = logging.getLogger(__name__)


def save_nodes(nodes) -> None:
    """Serialize nodes to JSON."""
    os.makedirs(os.path.dirname(NODE_CACHE_FILE), exist_ok=True)
    serialized = []
    for node in nodes:
        serialized.append({
            "node_id": node.node_id,
            "text": node.get_content(),
            "metadata": node.metadata,
        })
    with open(NODE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    log.info("Saved %d nodes to %s.", len(serialized), NODE_CACHE_FILE)


def load_cached_nodes():
    """Deserialize nodes from JSON. Returns None if cache missing."""
    if not os.path.exists(NODE_CACHE_FILE):
        # Try migrating legacy pickle
        legacy = os.path.join(NODE_CACHE_DIR, "nodes.pkl")
        if os.path.exists(legacy):
            import pickle
            log.warning("Migrating legacy pickle cache to JSON.")
            with open(legacy, "rb") as f:
                nodes = pickle.load(f)
            save_nodes(nodes)
            return nodes
        return None

    from llama_index.core.schema import TextNode
    with open(NODE_CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = []
    for item in data:
        node = TextNode(text=item["text"], metadata=item.get("metadata", {}))
        node.id_ = item["node_id"]
        nodes.append(node)

    log.info("Loaded %d nodes from JSON cache.", len(nodes))
    return nodes
