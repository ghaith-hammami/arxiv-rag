"""
Retrieves relevant chunks with dense search, BM25, RRF, and re-ranking.
"""

# 1. standard library
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

# 2. third-party
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# 3. local
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


@lru_cache(maxsize=1)
def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads the embedding model once.

    Returns:
        the embedding model
    """
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def load_chunk_documents() -> list[Document]:
    """
    Loads chunk documents from the saved JSON file.

    Returns:
        a list of LangChain documents
    """
    chunks_path = Path(config.CHUNKS_PATH)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {config.CHUNKS_PATH}. "
            "Did you run ingest.py first?"
        )

    with chunks_path.open("r", encoding="utf-8") as file:
        chunk_dicts = json.load(file)

    documents = []
    for chunk in chunk_dicts:
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "parent_id": chunk["parent_id"],
                    "parent_text": chunk["parent_text"],
                    "paper_id": chunk["paper_id"],
                    "title": chunk["title"],
                    "authors": chunk["authors"],
                    "published": chunk["published"],
                    "url": chunk["url"],
                },
            )
        )

    return documents


@lru_cache(maxsize=1)
def load_vector_store() -> FAISS:
    """
    Loads the saved FAISS index.

    Returns:
        the FAISS vector store
    """
    index_path = Path(config.FAISS_INDEX_PATH)
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {config.FAISS_INDEX_PATH}. "
            "Did you run ingest.py first?"
        )

    return FAISS.load_local(
        str(index_path),
        load_embedding_model(),
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def load_bm25_retriever() -> BM25Retriever:
    """
    Loads the BM25 retriever once.

    Returns:
        the configured BM25 retriever
    """
    retriever = BM25Retriever.from_documents(load_chunk_documents())
    retriever.k = config.TOP_K_SPARSE
    return retriever


@lru_cache(maxsize=1)
def load_cross_encoder() -> HuggingFaceCrossEncoder:
    """
    Loads the cross-encoder once.

    Returns:
        the configured cross-encoder
    """
    return HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)


def dense_retrieve(query: str, vector_store: FAISS) -> list[Document]:
    """
    Retrieves chunks with dense vector search.

    Args:
        query: user question
        vector_store: loaded FAISS vector store

    Returns:
        top dense-retrieved chunk documents
    """
    print("Running dense retrieval...")
    return vector_store.similarity_search(query, k=config.TOP_K_DENSE)


def sparse_retrieve(query: str, chunk_documents: list[Document]) -> list[Document]:
    """
    Retrieves chunks with BM25 keyword search.

    Args:
        query: user question
        chunk_documents: all chunk documents

    Returns:
        top BM25-retrieved chunk documents
    """
    print("Running sparse retrieval...")
    return load_bm25_retriever().invoke(query)


def fuse_results(
    dense_documents: list[Document],
    sparse_documents: list[Document],
) -> list[Document]:
    """
    Merges dense and sparse results with reciprocal rank fusion.

    Args:
        dense_documents: results from dense retrieval
        sparse_documents: results from sparse retrieval

    Returns:
        fused chunk documents sorted by RRF score
    """
    scores: dict[str, float] = {}
    documents_by_chunk_id: dict[str, Document] = {}

    for rank, document in enumerate(dense_documents, start=1):
        chunk_id = document.metadata["chunk_id"]
        documents_by_chunk_id[chunk_id] = document
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (config.RRF_K + rank)

    for rank, document in enumerate(sparse_documents, start=1):
        chunk_id = document.metadata["chunk_id"]
        documents_by_chunk_id[chunk_id] = document
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (config.RRF_K + rank)

    ranked_chunk_ids = sorted(scores, key=scores.get, reverse=True)
    return [documents_by_chunk_id[chunk_id] for chunk_id in ranked_chunk_ids]


def rerank_chunks(query: str, chunk_documents: list[Document]) -> list[Document]:
    """
    Re-ranks fused chunks with a cross-encoder.

    Args:
        query: user question
        chunk_documents: fused chunk documents

    Returns:
        top re-ranked chunk documents
    """
    print("Re-ranking fused results...")
    if not chunk_documents:
        return []

    # Sibling children return the same parent text. Score each parent once so
    # repeated passages cannot fill the final context or waste reranker work.
    seen_parent_ids = set()
    parent_documents = []

    for document in chunk_documents:
        parent_id = document.metadata["parent_id"]
        if parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            parent_documents.append(document)

    text_pairs = []
    for document in parent_documents:
        text_pairs.append((query, document.metadata["parent_text"]))

    raw_scores = load_cross_encoder().score(text_pairs)
    scored_documents = []

    for document, raw_score in zip(parent_documents, raw_scores):
        score = float(raw_score)
        document.metadata["reranker_score"] = score
        document.metadata["normalized_score"] = normalize_score(score)
        scored_documents.append(document)

    scored_documents.sort(
        key=lambda document: document.metadata["reranker_score"],
        reverse=True,
    )
    return scored_documents[: config.TOP_K_RERANK]


def normalize_score(score: float) -> float:
    """
    Normalizes a reranker score to a 0-1 range.

    Args:
        score: raw cross-encoder score

    Returns:
        normalized score between 0 and 1
    """
    if score >= 0:
        return 1 / (1 + math.exp(-score))

    exponent = math.exp(score)
    return exponent / (1 + exponent)


def retrieve_chunks(query: str, dense_query: str | None = None) -> list[Document]:
    """
    Runs the full retrieval pipeline.

    Args:
        query: standalone user question for keyword search and re-ranking
        dense_query: optional HyDE passage used only for dense search

    Returns:
        top chunk documents after re-ranking
    """
    chunk_documents = load_chunk_documents()
    vector_store = load_vector_store()

    dense_documents = dense_retrieve(dense_query or query, vector_store)
    sparse_documents = sparse_retrieve(query, chunk_documents)
    fused_documents = fuse_results(dense_documents, sparse_documents)
    return rerank_chunks(query, fused_documents)


def warm_retrieval_resources() -> None:
    """
    Warms cached retrieval resources for faster first queries.
    """
    print("Loading retrieval resources...")
    load_chunk_documents()
    load_vector_store()
    load_bm25_retriever()
    load_cross_encoder()


def clear_retrieval_caches() -> None:
    """
    Clears cached retrieval resources after new ingestion.
    """
    load_chunk_documents.cache_clear()
    load_vector_store.cache_clear()
    load_bm25_retriever.cache_clear()
    load_cross_encoder.cache_clear()
