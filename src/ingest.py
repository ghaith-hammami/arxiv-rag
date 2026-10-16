"""
Fetches ArXiv papers, chunks them, and builds a FAISS index.
"""

# 1. standard library
import argparse
import json
from pathlib import Path
import sys

# 2. third-party
import arxiv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# 3. local
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def fetch_papers(query: str, max_papers: int, category: str) -> list[dict]:
    """
    Fetches papers from ArXiv.

    Args:
        query: user topic used to search ArXiv
        max_papers: maximum number of papers to fetch
        category: ArXiv category filter such as "cs.LG"

    Returns:
        a list of paper dictionaries with text and metadata
    """
    print(f"Fetching {max_papers} papers from ArXiv for query: '{query}'...")

    papers = []
    search = arxiv.Search(
        query=f"all:{query} AND cat:{category}",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client(
        page_size=max_papers,
        delay_seconds=3.0,
        num_retries=5,
    )

    try:
        for result in client.results(search):
            papers.append(
                {
                    "paper_id": result.entry_id.rsplit("/", maxsplit=1)[-1],
                    "title": result.title,
                    "text": f"Title: {result.title}\n\nSummary: {result.summary}",
                    "authors": [author.name for author in result.authors],
                    "published": result.published.date().isoformat(),
                    "url": result.entry_id,
                }
            )
    except arxiv.HTTPError as error:
        raise RuntimeError(
            "ArXiv rate-limited the request. Wait a minute and try again, "
            "or use a more specific query."
        ) from error

    return papers


def chunk_papers(papers: list[dict]) -> list[Document]:
    """
    Splits papers into parent and child chunks.

    Args:
        papers: raw paper dictionaries from ArXiv

    Returns:
        child chunk documents ready for FAISS indexing
    """
    print("Chunking papers into parent and child documents...")

    tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
    parent_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    child_documents = []
    parent_number = 0
    child_number = 0

    for paper in papers:
        parent_chunks = parent_splitter.split_text(paper["text"])

        for parent_text in parent_chunks:
            parent_number += 1
            parent_id = f"parent_{parent_number:04d}"
            child_chunks = child_splitter.split_text(parent_text)

            for child_text in child_chunks:
                child_number += 1
                child_documents.append(
                    Document(
                        page_content=child_text,
                        metadata={
                            "chunk_id": f"child_{child_number:04d}",
                            "parent_id": parent_id,
                            "parent_text": parent_text,
                            "paper_id": paper["paper_id"],
                            "title": paper["title"],
                            "authors": paper["authors"],
                            "published": paper["published"],
                            "url": paper["url"],
                        },
                    )
                )

    return child_documents


def save_chunks(child_documents: list[Document]) -> None:
    """
    Saves child chunks to the configured JSON file.

    Args:
        child_documents: child chunk documents created during ingestion
    """
    output_path = Path(config.CHUNKS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    for document in child_documents:
        chunks.append(
            {
                "chunk_id": document.metadata["chunk_id"],
                "parent_id": document.metadata["parent_id"],
                "paper_id": document.metadata["paper_id"],
                "title": document.metadata["title"],
                "text": document.page_content,
                "parent_text": document.metadata["parent_text"],
                "authors": document.metadata["authors"],
                "published": document.metadata["published"],
                "url": document.metadata["url"],
            }
        )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2)

    print(f"Saved {len(chunks)} chunks to {output_path}.")


def build_index(child_documents: list[Document]) -> None:
    """
    Builds and saves the FAISS index.

    Args:
        child_documents: child chunk documents to index
    """
    print(f"Embedding {len(child_documents)} chunks. This may take a minute...")

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(child_documents, embeddings)

    index_path = Path(config.FAISS_INDEX_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))

    print(f"FAISS index saved to {index_path}.")


def run_ingestion(query: str, max_papers: int, category: str) -> None:
    """
    Runs the full ingestion pipeline.

    Args:
        query: user topic used to fetch papers
        max_papers: number of papers to fetch
        category: ArXiv category filter
    """
    papers = fetch_papers(query=query, max_papers=max_papers, category=category)
    if not papers:
        raise ValueError("No papers were fetched. Try a broader query.")

    child_documents = chunk_papers(papers)
    save_chunks(child_documents)
    build_index(child_documents)
    from src.retrieve import clear_retrieval_caches

    clear_retrieval_caches()
    print("Ingestion complete.")


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Ingest ArXiv papers into FAISS.")
    parser.add_argument("--query", required=True, help="Search query for ArXiv.")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=config.DEFAULT_MAX_PAPERS,
        help="Maximum number of papers to fetch.",
    )
    parser.add_argument(
        "--category",
        default=config.ARXIV_CATEGORY,
        help="ArXiv category such as cs.LG.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Runs ingestion from the command line.
    """
    args = parse_args()
    run_ingestion(
        query=args.query,
        max_papers=args.max_papers,
        category=args.category,
    )


if __name__ == "__main__":
    main()
