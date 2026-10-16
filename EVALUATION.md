# Evaluation

Run complete: **False**. Cases collected: 30/30.

Detailed scoring was stopped early. All 30 answers and retrieval traces were collected; the denominators below show which factual-answer metrics finished. This is a partial scoring run, not a fully completed benchmark.

The fixed development set contains 24 answerable and six unsupported or misleading questions covering 25 indexed arXiv abstracts. It includes direct facts, paraphrases, comparisons, limitations, follow-ups, missing details, a false premise, an out-of-scope request, and an instruction-conflict case. It is authored development data, not an independent held-out test.

## Component checks

- dataset_evidence_matches_index: PASS
- faiss_vector_count: PASS
- faiss_json_chunk_alignment: PASS
- embedding_dimension: PASS
- rrf_rewards_agreement: PASS
- parent_child_chunking: PASS
- blank_question: PASS
- clear_chat: PASS
- chat_history_window: PASS
- gradio_interface_builds: PASS
- live_arxiv_ingestion_fetch: PASS

## Retrieval ablations

Paper-level results on answerable cases; unsupported requests have no positive paper label and are excluded from retrieval averages. Full results also include cutoffs 1 and 3, Precision, average precision, nDCG, and per-question rankings.

- dense: Recall@5 **1.0000**, MRR@5 **0.9722**, nDCG@5 **0.9778**.
- bm25: Recall@5 **0.9514**, MRR@5 **0.9062**, nDCG@5 **0.8919**.
- hybrid: Recall@5 **1.0000**, MRR@5 **1.0000**, nDCG@5 **0.9986**.
- hybrid_reranked: Recall@5 **1.0000**, MRR@5 **0.9792**, nDCG@5 **0.9846**.
- hyde_hybrid_reranked: Recall@5 **1.0000**, MRR@5 **0.9792**, nDCG@5 **0.9846**.

## RAGAS scores

These are the installed RAGAS metric implementations. Few-shot examples are removed, judge output is compact JSON with short reasons, and answer-relevancy strictness is 1 to fit the free API quota. Each mean shows the scored/eligible count. Missing or failed metrics are recorded explicitly, never silently replaced by zero or perfect scores.

- answer_correctness: **0.7714** (21/24 cases).
- answer_relevancy: **0.8913** (21/24 cases).
- context_recall: **1.0000** (21/24 cases).
- faithfulness: **0.9515** (22/24 cases).
- id_based_context_precision: **0.2273** (22/24 cases).
- id_based_context_recall: **1.0000** (22/24 cases).
- semantic_similarity: **0.8546** (21/24 cases).

Context precision uses RAGAS **IDBasedContextPrecision**, against annotated source-paper IDs; it is not the separate LLM ContextPrecision metric. Context recall is measured both by IDs and by an LLM checking reference claims against retrieved text. Faithfulness checks response claims against context, correctness uses the default 75% factual F1 plus 25% semantic similarity against the reference, relevance uses a generated question and embedding similarity, and semantic similarity uses the existing BGE embedder.

## Unsupported questions and judge controls

- Correct refusal or premise correction: **1.0000** (6/6 judged).
- Faithfulness control `supported`: score 1.0000, expected 1.0, PASS.
- Faithfulness control `invented`: score 0.0000, expected 0.0, PASS.

Unsupported cases are judged separately; scoring a correct refusal as an ordinary factual answer would distort the core metric means. For these cases the judge checks refusal or premise correction and whether unsupported details were invented.

## Inspection examples

These are automated faithfulness flags, not independent human judgments. Full context and judge outputs remain in the JSON report.

- q02: Conventions reduce computational costs. Judge reason: Not explicitly stated.
- q05: Population size generates necessary diversity. Judge reason: Effective, not necessarily necessary.
- q05: Intrinsic motivation generates necessary diversity. Judge reason: Effective, not necessarily necessary.
- q07: AREL addresses agent redistribution. Judge reason: Addresses agent importance, not redistribution.
- q10: Second mechanism is agent deactivation Judge reason: Deactivation is part of mechanism one.

## Limits and audit trail

- Generator and judge use the same hosted Qwen model, so evaluator bias is correlated. These are automated judgments, not human-verified answer-accuracy labels.
- Source precision is relative to the annotated reference papers. Other retrieved background papers may be broadly related without being labeled as reference evidence.
- HyDE is compared with the same hybrid reranker using the standalone question. MRR improved in 0 cases, worsened in 0, and was unchanged in 24.
- Timings include deliberate free-tier waits; retrieval timing also includes the ablation work. They are not a production latency benchmark.
- API usage recorded, including pilots: 211 successful completions, 196676 tokens. Cache hits make no API requests.
- `data/evaluation_report.json` contains generated answers, contexts, source IDs, query rewrites, HyDE passages, per-case metrics/errors, usage, and raw judge outputs.
- `data/eval_questions.json` contains references, expected source IDs, category labels, history, and supporting evidence. `data/eval_corpus.json` pins all 25 source abstracts.

## Reproduce

Run from the project folder with the documented dependencies and GROQ_API_KEY.

```powershell
.\.venv\Scripts\python.exe evaluate.py --prepare-corpus
.\.venv\Scripts\python.exe evaluate.py --retrieval-only
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe evaluate.py --summarize-only
```

`--prepare-corpus` replaces the current generated index/chunks with the pinned evaluation corpus. The full run resumes matching checkpoints. `--limit 2` performs a pilot; `--fresh` deliberately starts a new checkpoint after settings or dataset changes. Free-tier rate limits can make a full run slow; no paid fallback is configured.

Metric definitions: [RAGAS documentation](https://docs.ragas.io/en/v0.4.3/concepts/metrics/available_metrics/).
