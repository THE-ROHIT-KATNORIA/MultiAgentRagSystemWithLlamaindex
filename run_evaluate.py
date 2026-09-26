"""
run_evaluate.py — Evaluation Suite Entry Point
===============================================

Usage:
    python run_evaluate.py
"""

import json
import os
import time
import logging
import re
from datetime import datetime

from config.settings import configure_llm
from database.memory import create_memory_table
from retrieval.vector_store import build_or_load_index
from retrieval.router import dynamic_route_and_filter
from retrieval.hybrid import create_hybrid_retriever
from engine.postprocessors import build_postprocessors
from engine.rag_engine import create_rag_engine
from llama_index.core import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

EVAL_DATASET = "./evaluation_dataset.json"
REPORT_MD    = "./evaluation_report.md"
REPORT_JSON  = "./evaluation_report_raw.json"
TOP_K        = 10


# ──────────────────────────────────────────────────────────────
# Retrieval Metrics
# ──────────────────────────────────────────────────────────────

def compute_retrieval_metrics(question, expected, index, nodes, reranker, sim_filter):
    decision = dynamic_route_and_filter(question)
    retriever, _ = create_hybrid_retriever(question, index, nodes, filters=decision["filters"])
    results = retriever.retrieve(question)[:TOP_K]

    key_phrases = [
        p.strip().lower()
        for p in expected.lower().split(".")
        if len(p.strip()) > 10
    ]
    relevant_ranks = [
        rank for rank, r in enumerate(results, 1)
        if any(kp in r.node.get_content().lower() for kp in key_phrases)
    ]
    n = len(results)
    return {
        "hit_rate": 1.0 if relevant_ranks else 0.0,
        "mrr": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "precision_at_k": len(relevant_ranks) / n if n else 0.0,
        "num_retrieved": n,
        "num_relevant": len(relevant_ranks),
    }


# ──────────────────────────────────────────────────────────────
# LLM-as-Judge Helpers
# ──────────────────────────────────────────────────────────────

def _judge(prompt: str, float_score: bool = False) -> dict:
    try:
        raw = Settings.llm.complete(prompt).text.strip()
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        parsed = json.loads(match.group() if match else raw)
        score = float(parsed.get("score", 0))
        if not float_score:
            score = int(score)
        return {"score": score, "reason": str(parsed.get("reason", ""))}
    except Exception as exc:
        return {"score": 0.0 if float_score else 0, "reason": str(exc)}


def judge_faithfulness(question, answer, context):
    return _judge(f"""Judge if the ANSWER is fully supported by the CONTEXT.
Score 1 if faithful, 0 if it contains information not in the context.
Reply with JSON: {{"score": <0 or 1>, "reason": "<brief>"}}

QUESTION: {question}
CONTEXT: {context[:3000]}
ANSWER: {answer}""")


def judge_relevancy(question, answer):
    return _judge(f"""Judge if the ANSWER directly addresses the QUESTION.
Score 1 if relevant, 0 if off-topic.
Reply with JSON: {{"score": <0 or 1>, "reason": "<brief>"}}

QUESTION: {question}
ANSWER: {answer}""")


def judge_correctness(question, answer, expected):
    return _judge(f"""Rate how correct the ACTUAL answer is vs the EXPECTED answer.
Score: 1.0=perfect, 0.75=mostly, 0.5=partial, 0.25=slight, 0.0=wrong.
Reply with JSON: {{"score": <0.0-1.0>, "reason": "<brief>"}}

QUESTION: {question}
EXPECTED: {expected}
ACTUAL: {answer}""", float_score=True)


# ──────────────────────────────────────────────────────────────
# Report Generator
# ──────────────────────────────────────────────────────────────

def generate_report(results: list, total_time: float) -> str:
    n = len(results)
    if n == 0:
        return "# Evaluation Report\n\nNo results.\n"

    def avg(key, sub=None):
        if sub:
            return sum(r[sub][key] for r in results) / n
        return sum(r[key] for r in results) / n

    a_hit  = avg("hit_rate", "retrieval")
    a_mrr  = avg("mrr", "retrieval")
    a_prec = avg("precision_at_k", "retrieval")
    a_faith = avg("score", "faithfulness")
    a_relev = avg("score", "relevancy")
    a_cor   = avg("score", "correctness")
    a_lat   = avg("latency_s")
    overall = (a_hit + a_mrr + a_faith + a_relev + a_cor) / 5
    grade = (
        "🟢 Excellent" if overall >= 0.8 else
        "🟡 Good"      if overall >= 0.6 else
        "🟠 Needs Improvement" if overall >= 0.4 else
        "🔴 Poor"
    )

    lines = [
        "# 📊 RAG Evaluation Report",
        f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Questions:** {n}  |  **Total time:** {total_time:.1f}s  |  **Avg latency:** {a_lat:.1f}s",
        "\n---\n",
        "## Summary\n",
        "| Metric | Score |", "|--------|-------|",
        f"| Hit Rate | {a_hit:.0%} |",
        f"| MRR | {a_mrr:.2f} |",
        f"| Precision@{TOP_K} | {a_prec:.0%} |",
        f"| Faithfulness | {a_faith:.0%} |",
        f"| Relevancy | {a_relev:.0%} |",
        f"| Answer Correctness | {a_cor:.0%} |",
        f"\n**Overall: {grade} ({overall:.0%})**\n",
        "\n---\n## Per-Question Results\n",
    ]

    for r in results:
        lines += [
            f"### Q{r['id']}: {r['question']}\n",
            f"**Route:** `{r['route']}` | **Dept:** `{r['department']}` | **Latency:** {r['latency_s']:.1f}s\n",
            f"**Expected:** {r['expected_answer']}\n",
            f"**Actual:** {r['actual_answer']}\n",
            "| Metric | Score | Note |", "|--------|-------|------|",
            f"| Hit Rate | {r['retrieval']['hit_rate']:.0%} | {r['retrieval']['num_relevant']}/{r['retrieval']['num_retrieved']} chunks |",
            f"| MRR | {r['retrieval']['mrr']:.2f} | — |",
            f"| Faithfulness | {r['faithfulness']['score']} | {r['faithfulness']['reason']} |",
            f"| Relevancy | {r['relevancy']['score']} | {r['relevancy']['reason']} |",
            f"| Correctness | {r['correctness']['score']:.0%} | {r['correctness']['reason']} |",
            "",
        ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(EVAL_DATASET):
        print(f"❌ {EVAL_DATASET} not found.")
        return

    with open(EVAL_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    configure_llm()
    create_memory_table()
    index, nodes = build_or_load_index()
    reranker, sim_filter = build_postprocessors()

    print(f"\n📊 Evaluation Suite — {len(dataset)} questions")
    print("=" * 60)

    results = []
    total_start = time.time()

    for i, item in enumerate(dataset, 1):
        qid, question, expected = item["id"], item["question"], item["expected_answer"]
        print(f"\n[{i}/{len(dataset)}] Q{qid}: {question}")
        t0 = time.time()

        try:
            decision = dynamic_route_and_filter(question)
            retrieval = compute_retrieval_metrics(question, expected, index, nodes, reranker, sim_filter)
            print(f"   Hit={retrieval['hit_rate']:.0%}  MRR={retrieval['mrr']:.2f}  P@{TOP_K}={retrieval['precision_at_k']:.0%}")

            engine, _ = create_rag_engine(question, index, nodes, reranker, sim_filter, filters=decision["filters"])
            response = engine.query(question)
            actual = str(response)
            context = "\n\n".join(n.node.get_content() for n in response.source_nodes) if hasattr(response, "source_nodes") else ""

            faith = judge_faithfulness(question, actual, context)
            relev = judge_relevancy(question, actual)
            corr  = judge_correctness(question, actual, expected)
            print(f"   Faith={faith['score']}  Relev={relev['score']}  Correct={corr['score']:.0%}")

            results.append({
                "id": qid, "question": question,
                "expected_answer": expected, "actual_answer": actual,
                "route": decision["route"], "department": decision["department"],
                "retrieval": retrieval,
                "faithfulness": faith, "relevancy": relev, "correctness": corr,
                "latency_s": time.time() - t0,
            })
        except Exception as exc:
            log.error("Q%d failed: %s", qid, exc)
            err = {"score": 0, "reason": str(exc)}
            results.append({
                "id": qid, "question": question,
                "expected_answer": expected, "actual_answer": f"ERROR: {exc}",
                "route": "ERROR", "department": "ERROR",
                "retrieval": {"hit_rate": 0, "mrr": 0, "precision_at_k": 0, "num_retrieved": 0, "num_relevant": 0},
                "faithfulness": err, "relevancy": err,
                "correctness": {"score": 0.0, "reason": str(exc)},
                "latency_s": time.time() - t0,
            })

    total_time = time.time() - total_start
    report = generate_report(results, total_time)

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done! Report: {REPORT_MD}")
    n = len(results)
    print(f"   Faithfulness: {sum(r['faithfulness']['score'] for r in results)/n:.0%}")
    print(f"   Relevancy:    {sum(r['relevancy']['score'] for r in results)/n:.0%}")
    print(f"   Correctness:  {sum(r['correctness']['score'] for r in results)/n:.0%}")


if __name__ == "__main__":
    main()
