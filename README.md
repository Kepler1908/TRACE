# trace_rag

Agentic RAG retrieval pipeline with planner decomposition, multi-tool search, deferred relevance judgment, and LLM-based reranking.

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure LLM

```bash
cp .env.example .env
```

Edit `.env` to set your provider and API key:

```bash
DEFAULT_LLM_PROVIDER=deepseek        # openai, anthropic, google, deepseek, openrouter, llama_cpp, vllm, sglang
DEFAULT_LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=sk-...
```

For local models (llama.cpp / vLLM / SGLang), set the base URL:

```bash
DEFAULT_LLM_PROVIDER=vllm
DEFAULT_LLM_MODEL=Qwen/Qwen3.5-27B
VLLM_BASE_URL=http://localhost:8000/v1
```

### 3. Prepare Your Corpus

Create two JSONL files:

**corpus_dataset.jsonl** — one document per line:

```jsonl
{"doc_id": "doc_001", "corpus": "newspapers", "date": "1887-01-15", "title": "Article title", "text_original": "Full document text..."}
{"doc_id": "doc_002", "corpus": "parliamentary", "date": "1887-01-16", "title": "Session title", "text_original": "Full document text..."}
```

Required fields: `doc_id`, `text_original`
Optional fields: `corpus`, `date`, `title`, `summary_brief`, `filename`, `contents_text`

**question_dataset.jsonl** — one question per line:

```jsonl
{"question_id": "q001", "question_type": "SH", "question": "What did X say about Y?", "gold_ids": ["doc_001"]}
```

Required fields: `question_id`, `question`, `gold_ids`
Optional: `question_type` (used for per-type metrics breakdown)

Place these files in `data/corpus/` (default path), or pass custom paths via `build_full_corpus(path=...)` and `load_questions_jsonl(path=...)`.

### 4. Build Embeddings (optional, for dense search channel)

```bash
python -m trace_rag.util.build_embeddings
```

This pre-computes embeddings and saves `embeddings_cache.npy` + `embeddings_doc_ids.json` in the corpus directory. Without this step, the pipeline still works using BM25 + date + grep channels.

### 5. Run

```bash
python -m trace_rag --limit 5 --seed 42
```

Key flags:

| Flag | Description |
|------|-------------|
| `--limit N` | Number of questions to evaluate |
| `--all` | Evaluate all answerable questions |
| `--type TYPE` | Filter by question type |
| `--seed N` | Random seed for question sampling |
| `--quiet` | Suppress per-step output |
| `--quick` | Run 1 question only (shortcut for `--limit 1`) |
| `--json-out FILE` | Save detailed per-question JSON report |
| `--resume` | Skip questions already in `--json-out` file |
| `--concurrency N` | Parallel question processing (default: 10) |
| `--with-narrative` | Generate narrative answers (costs ~28% more tokens) |
| `--thinking` | Enable thinking mode (sets all token caps to 65536) |
| `--max-steps N` | Max agent search steps per sub-question (default: 13) |
| `--max-subquestions N` | Max sub-questions per question (default: 4) |

---

## Architecture

```
Question
  |
Planner LLM ── decompose into sub-questions with corpus targets
  |
For each sub-question (sequential, with briefing + re-planning):
  Phase 1: 3-channel RRF fusion (BM25 + date/metadata + dense embedding)
  Phase 2: Agent loop (search + review cycle with structured memory)
    Tools: search_bm25, search_date, search_semantic, search_grep, finish
    Decisions: accept / hold / reject per document chunk
  Post-agent: held re-evaluation + briefing generation
  |
  Briefing → Re-planner (can modify or spawn new sub-questions)
  |
Merge sub-question results → LLM rerank → final candidate list
```

---

## Adapting to Your Corpus

### Prompts (`src/prompts.py`)

The pipeline's behavior is driven by system prompts. To adapt to a new corpus:

```python
# src/prompts.py — top of file
CORPUS_LANG = "French"                    # language of your documents
AVAILABLE_CORPORA = ["parliamentary", "le_gaulois", "l_intransigeant"]
CORPUS_DESCRIPTION = "19th-century French parliamentary debates and newspaper articles"
```

- `CORPUS_LANG` — ensures the LLM preserves entity names in the source language
- `AVAILABLE_CORPORA` — the planner uses this to assign sub-questions to corpora
- `CORPUS_DESCRIPTION` — injected into system prompts so the LLM understands the domain

Prompts that are used directly (`AGENT_SYSTEM`, `RERANK_SYSTEM`, `NARRATIVE_SYSTEM`, etc.) are pre-formatted at import time via `_sub()`. Prompts with runtime placeholders (`PLANNER_SYSTEM`, `REPLAN_SYSTEM`, `BRIEFING_SYSTEM`) are `.replace()`-ed at call sites.

You can add domain-specific knowledge to the prompts to improve performance. For example, if your corpus has known relationships between sub-corpora (e.g., newspapers reporting on parliamentary events), adding this as guidance in `AGENT_SYSTEM` helps the agent make better search decisions.

### Search Tools

The built-in search tools (`search_bm25`, `search_date`, `search_grep`, `search_semantic`) cover most use cases. If your corpus has rich metadata beyond date and corpus name, consider adding a custom search tool in `src/search/` (e.g., a metadata facet search).

### Evaluation

To run evaluation with gold annotations:

1. Include `gold_ids` in your questions JSONL — these are `doc_id` values that answer the question
2. The pipeline computes recall@k, MRR, and precision/recall in the agent's accepted set
3. `--json-out results.json` saves full diagnostics per question for offline analysis

---

## Tuning & Debugging

### Token Budgets (`src/types.py`)

All LLM call token limits are defined as constants and in `AgentLimits`:

```python
AGENT_SEARCH_MAX_TOKENS = 768     # per agent search step
AGENT_REVIEW_MAX_TOKENS = 1536    # per agent review step
PLANNER_MAX_TOKENS = 2048         # planner decomposition
RERANK_MAX_TOKENS = 1024          # reranking
BRIEFING_MAX_TOKENS = 1024        # post-subquestion briefing
REPLAN_MAX_TOKENS = 1024          # re-planning between sub-questions
```

For local models with larger context windows, `LOCAL_AGENT_LIMITS` applies automatically (detected via provider name). For thinking models, `--thinking` overrides all caps to 65536.

Adjust these directly in `src/types.py` when debugging or tuning for a specific model.

### Pipeline Parameters (`src/types.py`)

```python
SUBQ_MAX_STEPS = 13         # max search steps per sub-question
MAX_SUB_QUESTIONS = 4       # max sub-questions per question
PHASE1_TOP_K_RATIO = 0.05   # Phase 1 candidate ratio
PHASE1_TOP_K_MIN = 20       # minimum Phase 1 candidates
RRF_K = 60                  # RRF smoothing constant
REVIEW_BATCH_SIZE = 8       # documents shown per review step
```

### Parse Error Handling

LLM output (especially from smaller or local models) can be unstable. The agent retries on JSON parse failures up to `AgentLimits.max_parse_errors` (default: 5, local: 2). If you see frequent parse errors, consider:
- Increasing `max_parse_errors` in `src/types.py`
- Switching to a model with more reliable structured output
- Enabling `--thinking` mode for models that support it

### Data Paths (`src/types.py`)

Default paths point to `data/rag_corpus/histoQA_thirdrepublic/`. Override them by passing explicit paths to `build_full_corpus()` and `load_questions_jsonl()`, or modify the constants directly.

---

## Project Structure

```
trace_rag/
  src/
    cli.py          # CLI entry point
    pipeline.py     # Orchestrates the 4-phase pipeline
    agent.py        # Phase 2 agent loop (search + review)
    prompts.py      # All LLM system prompts (configurable)
    scoring.py      # Phase 1 RRF fusion
    types.py        # Dataclasses, constants, token limits
    data.py         # JSONL loading + chunking
    tools.py        # Unified interface (re-exports + helpers)
    metrics.py      # IR evaluation metrics
    search/
      bm25.py       # BM25 keyword search
      cosine.py     # Dense embedding search
      grep.py       # Exact substring match
      date.py       # Date/date-range search
  util/
    llm_client.py   # Multi-provider LLM client
    config.py       # Pydantic settings (.env loading)
    build_embeddings.py  # Pre-compute corpus embeddings
    text.py         # Tokenization utilities
    logging.py      # Structured logging
```

## License

MIT
