import os
from pathlib import Path


def load_env_file() -> None:
    """
    Loads values from a local .env file into the environment.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        if "=" not in stripped_line:
            continue

        key, value = stripped_line.split("=", maxsplit=1)
        cleaned_key = key.strip()
        cleaned_value = value.strip().strip('"').strip("'")

        if cleaned_key and cleaned_key not in os.environ:
            os.environ[cleaned_key] = cleaned_value


load_env_file()


# Hugging Face model used to embed paper chunks and retrieval queries.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Cross-encoder model used to re-rank retrieved chunks.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Hosted Groq model used for HyDE, question rewriting, and final answers.
GENERATOR_MODEL = "qwen/qwen3.8-27b"

# Environment variable containing the Groq free-tier API key.
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"


# Number of tokens in large parent chunks returned to the generator.
PARENT_CHUNK_SIZE = 512

# Number of tokens in smaller child chunks stored in the vector index.
CHILD_CHUNK_SIZE = 128

# Token overlap between adjacent chunks to reduce context loss.
CHUNK_OVERLAP = 20


# Number of dense retrieval results to fetch before fusion.
TOP_K_DENSE = 20

# Number of sparse BM25 results to fetch before fusion.
TOP_K_SPARSE = 20

# Number of chunks kept after cross-encoder re-ranking.
TOP_K_RERANK = 5

# RRF constant that softens the gap between very high ranks.
RRF_K = 60


# Whether to expand the user query with a hypothetical answer.
USE_HYDE = True

# Number of papers to fetch from ArXiv during ingestion by default.
DEFAULT_MAX_PAPERS = 20

# Default ArXiv category for machine learning papers.
ARXIV_CATEGORY = "cs.LG"

# Number of recent chat messages to keep in prompts.
CHAT_HISTORY_WINDOW = 6

# Default paper count for topic-driven notebook ingestion.
NOTEBOOK_MAX_PAPERS = 25


# Folder where the FAISS index will be stored on disk.
FAISS_INDEX_PATH = "data/faiss_index"

# JSON file where child chunks and their metadata will be stored.
CHUNKS_PATH = "data/papers/chunks.json"

# JSON file with evaluation questions and reference answers.
EVAL_DATA_PATH = "data/eval_questions.json"

# Saved paper-level retrieval metrics from the small development benchmark.
RETRIEVAL_METRICS_PATH = "data/retrieval_metrics.json"

# Complete, resumable stage evaluation and the fixed reference corpus.
EVALUATION_REPORT_PATH = "data/evaluation_report.json"
EVALUATION_CORPUS_PATH = "data/eval_corpus.json"
EVALUATION_SUMMARY_PATH = "EVALUATION.md"
EVALUATION_CACHE_PATH = ".eval_cache"

# Keep evaluation within Groq's free token limits; never upgrade billing.
EVALUATION_TOKENS_PER_MINUTE = 6500
EVALUATION_OUTPUT_TOKENS_PER_MINUTE = 900
EVALUATION_TOKEN_BUDGET = 220000
EVALUATION_JUDGE_MAX_TOKENS = 1536
EVALUATION_OUTPUT_RESERVE = 192
EVALUATION_REASON_WORD_LIMIT = 5
EVALUATION_CHARS_PER_TOKEN = 3
EVALUATION_RATE_WINDOW_SECONDS = 60
EVALUATION_SLEEP_SECONDS = 30
EVALUATION_RATE_RETRIES = 3

# Retrieval metric cutoffs, kept explicit in the saved report.
EVALUATION_CUTOFFS = (1, 3, 5)


# Maximum number of tokens to generate for HyDE responses.
HYDE_MAX_NEW_TOKENS = 256

# Maximum number of tokens to generate for final answers.
ANSWER_MAX_NEW_TOKENS = 512

# Low temperature keeps answers more focused and deterministic.
GENERATION_TEMPERATURE = 0.2

# Fail clearly if the hosted generator does not respond in time.
GENERATION_TIMEOUT = 45

# Retry one transient API failure without repeatedly consuming free quota.
GENERATION_MAX_RETRIES = 1


# Environment variable name for the Hugging Face API token.
HUGGINGFACE_TOKEN_ENV_VAR = "HUGGINGFACEHUB_API_TOKEN"

# Read the token directly from the environment when needed.
HUGGINGFACE_API_TOKEN = os.getenv(HUGGINGFACE_TOKEN_ENV_VAR)

# Mirror the token into HF_TOKEN so Hugging Face libraries use the same auth.
if HUGGINGFACE_API_TOKEN and "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = HUGGINGFACE_API_TOKEN

if not HUGGINGFACE_API_TOKEN and "HF_TOKEN" in os.environ:
    HUGGINGFACE_API_TOKEN = os.environ["HF_TOKEN"]
    os.environ[HUGGINGFACE_TOKEN_ENV_VAR] = HUGGINGFACE_API_TOKEN
