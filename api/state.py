"""
Application State
=================

Shared startup state (index, nodes, reranker, agent)
initialised once at lifespan and reused across all requests.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppState:
    index: Optional[object] = None
    nodes: Optional[list] = None
    reranker: Optional[object] = None
    sim_filter: Optional[object] = None
    agent: Optional[object] = None
    ready: bool = False


# Singleton — imported by routers
state = AppState()
