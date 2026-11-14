"""
Document Loader & Chunker
=========================

Loads documents from disk, splits into chunks,
and assigns department metadata.
"""

import logging
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter

from config.settings import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

log = logging.getLogger(__name__)


def assign_department(file_name: str) -> str:
    """Map a source filename to a department label based on word tokens."""
    import re
    tokens = set(re.split(r'[^a-zA-Z0-9]+', file_name.lower()))

    if tokens & {"finance", "budget", "payroll", "expense", "invoice", "financial"}:
        return "Finance"
    if tokens & {"security", "infosec", "cyber", "infrastructure", "network"} or "it" in tokens:
        return "IT"
    if tokens & {"hr", "leave", "employee", "resume", "cv", "hiring", "recruitment", "profile"}:
        return "HR"

    return "NONE"


def load_and_chunk_documents():
    """Load documents from DATA_DIR, chunk, and assign department metadata.

    Returns:
        list[TextNode]: Chunked nodes with metadata.
    """
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    log.info("Loaded %d documents from %s.", len(documents), DATA_DIR)

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    Settings.node_parser = splitter
    nodes = splitter.get_nodes_from_documents(documents)

    for node in nodes:
        file_name = str(node.metadata.get("file_name", ""))
        node.metadata["department"] = assign_department(file_name)

    log.info("Created %d nodes.", len(nodes))
    return nodes
