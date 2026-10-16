# ArXiv RAG

A conversational RAG app for exploring AI/ML research abstracts, with hybrid
retrieval, source links, and a measured evaluation of the pipeline.

It fetches titles and abstracts from arXiv, indexes them locally, and uses a
hosted Qwen model through Groq to answer questions. Full paper PDFs are not
indexed. This is a portfolio project with documented limitations.

## How it works

1. Fetch abstracts and split them into parent and child chunks.
2. Retrieve child chunks with FAISS dense search and BM25 keyword search.
3. Combine rankings with Reciprocal Rank Fusion and rerank distinct parent passages.
4. Generate an answer from the retrieved evidence and show the source papers.

Follow-up questions are rewritten using recent chat history. Optional HyDE
creates a hypothetical passage for dense search; BM25 and the reranker keep
using the standalone question. The Gradio interface displays the conversation,
source links, and retrieval relevance.

**Stack:** Python, LangChain, FAISS, BM25, Sentence Transformers, Groq,
Gradio, and RAGAS. Embeddings use `BAAI/bge-small-en-v1.5`, reranking uses
`cross-encoder/ms-marco-MiniLM-L-6-v2`, and generation uses
`qwen/qwen3.8-27b` through Groq.

## Quick start

Tested with **Python 3.13**. The commands below use PowerShell.

```powershell
git clone https://github.com/ghaith-hammami/arxiv-rag.git
cd arxiv-rag
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your free-tier `GROQ_API_KEY` to `.env`. Create a key at
[Groq Console](https://console.groq.com/keys). A Hugging Face token is optional
for public model downloads. Real keys belong only in your local `.env` or
environment variables.

Build the sample index from the included 25-abstract corpus and launch the app:

```powershell
.\.venv\Scripts\python.exe evaluate.py --prepare-corpus
.\.venv\Scripts\python.exe app.py
```

Open the local URL printed by Gradio. For example, ask:

- What makes Pommerman challenging for multi-agent learning?
- What kinds of datasets does OG-MARL provide?
- What exact learning rate did AREL use? (The abstracts do not contain this detail.)

Embedding and reranking models download on first use and run locally. Generation
runs on Groq; no local generation model or paid inference endpoint is required.
Groq's free tier has quotas, and model availability can change. See
[Groq rate limits](https://console.groq.com/docs/rate-limits). Set
`USE_HYDE = False` in `config.py` to use fewer generation requests. If the app
returns HTTP 429, wait for your quota to reset.

## Index a different topic

```powershell
.\.venv\Scripts\python.exe src/ingest.py --query "multi-agent reinforcement learning" --max-papers 25 --category cs.LG
```

This replaces the generated chunks and FAISS index. Run commands from the
project root. The index is generated locally and is not committed.

You can also use the pipeline directly:

```python
from src.pipeline import run

result = run("What makes Pommerman challenging?")
print(result["answer"])
print(result["sources"])
```

## Evaluation results

The dataset contains **30 authored questions** covering all 25 pinned abstracts:
24 answerable cases and six unsupported or misleading requests. It includes
facts, paraphrases, limitations, comparisons, follow-ups, missing details,
a false premise, an out-of-scope request, and an instruction conflict.

All 30 answers and retrieval traces were collected. Detailed RAGAS scoring
was stopped early: 21 answerable cases have all seven scores, and one has
partial scores. The remaining two have retrieval and answer traces only.

Measured results from this partial run:

- Hybrid + reranking + HyDE: **Recall@5 1.0000**, **MRR@5 0.9792** on 24 cases.
- Faithfulness: **0.9515** on 22/24 eligible cases.
- Answer relevancy: **0.8913** on 21/24 cases.
- Answer correctness: **0.7714** on 21/24 cases.
- LLM context recall: **1.0000** on 21/24 cases.
- ID-based context precision: **0.2273** on 22/24 cases.
- Unsupported-request handling: **6/6 passed**.
- **11 component checks passed**, including live ingestion, chunking, index
  alignment, embedding dimensions, rank fusion, chat controls, and UI construction.

[EVALUATION.md](EVALUATION.md) includes all metric means, scoring coverage,
retrieval ablations, judge controls, and limitations.
[data/evaluation_report.json](data/evaluation_report.json) preserves the
per-question answers, actual context, rankings, rewrites, metrics, and judge outputs.

Run the API-free retrieval comparison:

```powershell
.\.venv\Scripts\python.exe evaluate.py --retrieval-only
```

Run a new full evaluation, or a two-question pilot:

```powershell
.\.venv\Scripts\python.exe evaluate.py --fresh
.\.venv\Scripts\python.exe evaluate.py --fresh --limit 2
```

These commands replace the saved evaluation report. Without `--fresh`, the
full evaluator resumes a matching checkpoint. It paces requests and caches
successful judge steps; a full free-tier run can be slow. Use the pinned
corpus for comparison, since a new arXiv search may return different papers.

## Limitations

- Answers use indexed titles and abstracts, so detailed methods and results
  from full papers may be unavailable.
- This small authored development set is not an independent held-out benchmark.
- The same hosted model generates and judges answers. Automated scores can be
  biased and are not human-verified accuracy labels.
- Context precision uses the RAGAS **ID-based** metric against labeled source
  papers. Returning five papers often includes additional background sources.
- HyDE did not improve MRR in this run; the results do not establish that every
  pipeline stage improves over simpler retrieval.
- Retrieval relevance is a normalized reranker signal, not a probability that
  the answer is correct. Timings include evaluation work and quota waits.

## Files

- `app.py`, `config.py`, `src/`: interface and RAG pipeline.
- `evaluate.py`: component checks, retrieval comparisons, and RAGAS scoring.
- `data/eval_corpus.json`: fixed abstracts with authors and arXiv links.
- `data/eval_questions.json`: questions, references, source labels, and evidence.
- `data/evaluation_report.json`, `EVALUATION.md`: saved results and interpretation.
- `.env.example`, `requirements.txt`: configuration template and tested dependencies.

Local keys, environments, caches, generated indexes, exploratory notebooks,
and personal development notes are excluded from Git.

## License

Project code is provided under the [MIT License](LICENSE). Research abstracts
retain their original authorship and rights; their authors and arXiv links
are included in the corpus. The project license does not relicense those texts.
