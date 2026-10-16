"""
Generates HyDE text and final answers from retrieved chunks.
"""

# 1. standard library
from functools import lru_cache
import os
from pathlib import Path
import sys

# 2. third-party
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# 3. local
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


@lru_cache(maxsize=2)
def load_generator_model(max_new_tokens: int) -> ChatGroq:
    """
    Creates a cached client for Groq's hosted generation model.

    Args:
        max_new_tokens: maximum number of tokens to generate

    Returns:
        the configured generator model
    """
    api_token = os.getenv(config.GROQ_API_KEY_ENV_VAR)
    if not api_token:
        raise ValueError(
            f"Missing {config.GROQ_API_KEY_ENV_VAR}. "
            "Set it in the environment or the project's .env file."
        )

    return ChatGroq(
        model=config.GENERATOR_MODEL,
        api_key=api_token,
        max_tokens=max_new_tokens,
        temperature=config.GENERATION_TEMPERATURE,
        timeout=config.GENERATION_TIMEOUT,
        max_retries=config.GENERATION_MAX_RETRIES,
    )


def select_recent_history(chat_history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """
    Keeps only the most recent chat messages.

    Args:
        chat_history: full chat history

    Returns:
        recent chat messages
    """
    if not chat_history:
        return []

    return chat_history[-config.CHAT_HISTORY_WINDOW :]


def build_history_messages(
    chat_history: list[dict[str, str]] | None,
) -> list[HumanMessage | AIMessage]:
    """
    Converts stored chat history into LangChain messages.

    Args:
        chat_history: stored chat messages

    Returns:
        LangChain message objects
    """
    messages: list[HumanMessage | AIMessage] = []

    for turn in select_recent_history(chat_history):
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    return messages


def rewrite_question_with_history(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Rewrites a follow-up question into a standalone search query.

    Args:
        query: latest user question
        chat_history: previous chat turns

    Returns:
        standalone question for retrieval
    """
    if not chat_history:
        return query

    print("Rewriting question with chat history...")

    messages = [
        SystemMessage(
            content=(
                "Rewrite the user's latest message into a standalone academic "
                "search question. Keep it concise. Preserve important names "
                "and technical terms. Return only the rewritten question."
            )
        )
    ]
    messages.extend(build_history_messages(chat_history))
    messages.append(HumanMessage(content=query))

    model = load_generator_model(config.HYDE_MAX_NEW_TOKENS)
    response = model.invoke(messages)
    rewritten_query = response.content.strip()

    if not rewritten_query:
        return query

    return rewritten_query


def generate_hyde_query(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Generates a hypothetical answer for retrieval.

    Args:
        query: user question
        chat_history: previous chat turns

    Returns:
        generated HyDE text
    """
    print("Generating HyDE query...")

    standalone_query = rewrite_question_with_history(query, chat_history)
    model = load_generator_model(config.HYDE_MAX_NEW_TOKENS)
    messages = [
        SystemMessage(
            content=(
                "Write a short academic-style passage that answers the user's "
                "question. Sound like a research paper, not a chatbot."
            )
        ),
        HumanMessage(content=standalone_query),
    ]
    response = model.invoke(messages)
    return response.content.strip()


def format_context(chunks: list[Document]) -> str:
    """
    Turns retrieved chunks into one context string.

    Args:
        chunks: retrieved chunk documents

    Returns:
        formatted context text
    """
    context_parts = []

    for chunk in chunks:
        context_parts.append(
            "\n\n".join(
                [
                    f"Title: {chunk.metadata['title']}",
                    f"Text: {chunk.metadata['parent_text']}",
                ]
            )
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    query: str,
    chunks: list[Document],
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Generates the final answer from retrieved chunks.

    Args:
        query: user question
        chunks: top retrieved chunk documents
        chat_history: previous chat turns

    Returns:
        generated answer text
    """
    print("Generating final answer...")

    model = load_generator_model(config.ANSWER_MAX_NEW_TOKENS)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful research assistant. Use the recent chat history "
                "only to understand references like 'it' or 'that paper'. "
                "Answer the user's latest question using only the provided context. "
                "If the answer is not in the context, say you do not have enough "
                "information."
            )
        )
    ]
    messages.extend(build_history_messages(chat_history))
    messages.append(
        HumanMessage(
            content=(
                f"Context:\n{format_context(chunks)}\n\n"
                f"Question:\n{query}"
            )
        )
    )
    response = model.invoke(messages)
    return response.content.strip()


def warm_generator_resources() -> None:
    """
    Warms cached generator resources for faster first queries.
    """
    print("Loading generation resources...")
    load_generator_model(config.HYDE_MAX_NEW_TOKENS)
    load_generator_model(config.ANSWER_MAX_NEW_TOKENS)


def clear_generator_caches() -> None:
    """
    Clears cached generator resources.
    """
    load_generator_model.cache_clear()
