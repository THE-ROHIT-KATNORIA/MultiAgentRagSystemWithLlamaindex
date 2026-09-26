"""
Sub-Question Query Engine
=========================

Wraps the RAG engine in a SubQuestionQueryEngine for
multi-part / comparative questions.
"""

from llama_index.core import Settings
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.question_gen import LLMQuestionGenerator
from engine.rag_engine import create_rag_engine


def create_sub_question_engine(question: str, index, nodes, reranker,
                                similarity_filter, filters=None):
    """Wrap a filtered RAG engine in a SubQuestionQueryEngine.

    Returns:
        (SubQuestionQueryEngine, MetadataFilters | None)
    """
    rag_engine, filters = create_rag_engine(
        question, index, nodes, reranker, similarity_filter, filters=filters
    )

    tool = QueryEngineTool.from_defaults(
        query_engine=rag_engine,
        description=(
            "Search the uploaded company documents for policies, "
            "rules, procedures, employee information and other "
            "company-specific information."
        ),
    )

    question_gen = LLMQuestionGenerator.from_defaults(llm=Settings.llm)

    sub_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[tool],
        question_gen=question_gen,
        llm=Settings.llm,
        use_async=False,
        verbose=True,
    )
    return sub_engine, filters
