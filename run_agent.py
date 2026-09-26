"""
run_agent.py — Agent Entry Point
=================================

Usage:
    python run_agent.py
"""

import asyncio
import logging

from config.settings import configure_llm
from database.memory import create_memory_table
from retrieval.vector_store import build_or_load_index
from engine.postprocessors import build_postprocessors
from tools.rag_tool import build_rag_tool
from tools import ALL_TOOLS
from agent.agent import build_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


async def run():
    # ── Startup ──
    configure_llm()
    create_memory_table()
    index, nodes = build_or_load_index()
    reranker, sim_filter = build_postprocessors()

    # Inject RAG components into the RAG tool
    build_rag_tool(index, nodes, reranker, sim_filter)

    agent = build_agent(ALL_TOOLS)

    # ── Welcome ──
    print("\n" + "=" * 60)
    print("🤖  COMPANY ASSISTANT AGENT")
    print("=" * 60)
    print("\nAvailable tools:")
    for tool in ALL_TOOLS:
        print(f"  • {tool.metadata.name}")
    print("\nType 'exit' to quit.\n")

    # ── Loop ──
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit", "bye"):
            print("\n👋 Goodbye!")
            break
        if not question:
            continue

        try:
            response = await agent.run(question)
            print(f"\n🤖 Agent: {response}")
        except Exception as exc:
            log.error("Agent error: %s", exc, exc_info=True)
            print(f"\n⚠  {exc}")

        print("\n" + "-" * 60)


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
