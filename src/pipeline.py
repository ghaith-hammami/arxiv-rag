"""
Connects retrieval and generation into one RAG pipeline.
"""

# 1. standard library
from pathlib import Path
import sys
from typing import Any

# 2. third-party
from langchain_core.documents import Document

# 3. local
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.generate import generate_answer, generate_hyde_query, rewrite_question_with_history
from src.retrieve import retrieve_chunks


def choose_retrieval_query(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Chooses the text used for dense retrieval.

    Args:
        query: original user question
        chat_history: previous chat turns

    Returns:
        the question or HyDE passage to send into dense retrieval
    """
    if config.USE_HYDE:
        return generate_hyde_query(query, chat_history)

    return rewrite_question_with_history(query, chat_history)


def build_sources(chunks: list[Document]) -> list[dict[str, str]]:
    """
    Builds a unique source list from retrieved chunks.

    Args:
        chunks: retrieved chunk documents

    Returns:
        source metadata for the final response
    """
    seen_paper_ids = set()
    sources = []

    for chunk in chunks:
        paper_id = chunk.metadata["paper_id"]
        if paper_id in seen_paper_ids:
            continue

        seen_paper_ids.add(paper_id)
        sources.append(
            {
                "title": chunk.metadata["title"],
                "arxiv_id": paper_id,
                "url": chunk.metadata["url"],
            }
        )

    return sources


def build_confidence(chunks: list[Document]) -> float:
    """
    Builds a confidence score from reranker scores.

    Args:
        chunks: retrieved chunk documents

    Returns:
        a confidence score between 0 and 1
    """
    if not chunks:
        return 0.0

    total_score = 0.0

    for chunk in chunks:
        total_score += chunk.metadata.get("normalized_score", 0.0)

    return total_score / len(chunks)


def run(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Runs the full RAG pipeline for one user question.

    Args:
        query: user question
        chat_history: previous chat turns

    Returns:
        final answer, sources, and confidence
    """
    print("Starting pipeline...")

    # HyDE expands semantic search; keep the actual question for keyword
    # matching and relevance scoring so hypothetical details cannot replace it.
    standalone_query = rewrite_question_with_history(query, chat_history)
    dense_query = choose_retrieval_query(standalone_query)
    retrieved_chunks = retrieve_chunks(standalone_query, dense_query=dense_query)
    answer = generate_answer(query, retrieved_chunks, chat_history)

    return {
        "answer": answer,
        "sources": build_sources(retrieved_chunks),
        "confidence": build_confidence(retrieved_chunks),
    }
