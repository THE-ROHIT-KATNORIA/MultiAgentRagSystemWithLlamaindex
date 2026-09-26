"""
run_rag.py — RAG Chat Loop Entry Point
=======================================

Usage:
    python run_rag.py
"""

import logging
from config.settings import configure_llm, MAX_INPUT_LENGTH
from database.memory import create_memory_table, save_message, get_chat_history
from retrieval.vector_store import build_or_load_index
from retrieval.router import dynamic_route_and_filter
from engine.postprocessors import build_postprocessors
from engine.rag_engine import create_rag_engine
from engine.sub_question import create_sub_question_engine
from engine.query_rewriter import rewrite_query
from llama_index.core import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


def print_sources(response) -> None:
    if not hasattr(response, "source_nodes") or not response.source_nodes:
        return
    print("\nSOURCES:")
    for i, source in enumerate(response.source_nodes, 1):
        node = source.node
        meta = node.metadata
        print(f"\n  [{i}] {meta.get('file_name', 'Unknown')} "
              f"| page {meta.get('page_label', meta.get('page_number', '?'))} "
              f"| score {source.score}")
        print(f"      {node.get_content()[:200]}...")


def main():
    # ── Startup ──
    configure_llm()
    create_memory_table()
    index, nodes = build_or_load_index()
    reranker, sim_filter = build_postprocessors()

    # Dept stats
    dept_counts: dict[str, int] = {}
    for n in nodes:
        d = n.metadata.get("department", "NONE")
        dept_counts[d] = dept_counts.get(d, 0) + 1
    log.info("Department distribution: %s", dept_counts)

    # ── Session ──
    session_id = input("\nEnter session ID (blank = 'default'): ").strip() or "default"
    print(f"Session: {session_id}")
    print("\n🔍 Advanced RAG ready — type 'exit' to quit.\n")

    # ── Chat loop ──
    while True:
        question = input("You: ").strip()

        if question.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if len(question) > MAX_INPUT_LENGTH:
            print(f"⚠  Input too long ({len(question)} chars, max {MAX_INPUT_LENGTH}).")
            continue

        try:
            history = get_chat_history(session_id=session_id, limit=20)
            rewritten = rewrite_query(question, history)
            print(f"\nRewritten: {rewritten}")

            decision = dynamic_route_and_filter(rewritten)
            route = decision["route"]
            filters = decision["filters"]

            print(f"\n🧠 Route: {route}  |  Dept: {decision['department']}"
                  f"  |  Confidence: {decision['confidence']:.0%}")
            print(f"   Reasoning: {decision['reasoning']}")

            response_text = ""

            if route == "RAG":
                engine, filters = create_rag_engine(
                    rewritten, index, nodes, reranker, sim_filter,
                    filters=filters, streaming=True,
                )
                print("\nAnswer:")
                streaming = engine.query(rewritten)
                for token in streaming.response_gen:
                    print(token, end="", flush=True)
                    response_text += token
                print()
                print_sources(streaming)

            elif route == "COMPLEX_RAG":
                engine, filters = create_sub_question_engine(
                    rewritten, index, nodes, reranker, sim_filter, filters=filters
                )
                response = engine.query(rewritten)
                response_text = str(response)
                print(f"\nAnswer:\n{response_text}")
                print_sources(response)

            else:  # DIRECT_LLM
                prompt = (
                    f"Answer the user's question directly. "
                    f"Do not mention company documents.\n\nQuestion: {question}"
                )
                print("\nAnswer:")
                for token in Settings.llm.stream_complete(prompt):
                    print(token.delta, end="", flush=True)
                    response_text += token.delta
                print()

            save_message(session_id, "user", question)
            save_message(session_id, "assistant", response_text)
            print("\n✅ Saved to PostgreSQL.")

        except Exception as exc:
            log.error("Error: %s", exc, exc_info=True)
            print(f"\n⚠  {exc}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
