# NASA Mission Intelligence Chat (RAG + RAGAS)

A Retrieval-Augmented Generation (RAG) system that answers plain-English questions about
**Apollo 11**, **Apollo 13** and **Challenger (STS-51L)**. It uses NASA mission transcripts
and technical documents, and scores every answer in real time with RAGAS.

```
data_text/*.txt ──► embedding_pipeline.py ──► ChromaDB (OpenAI embeddings + metadata)
                                                   │
user question ──► rag_client.py (semantic search, mission filter, dedup, context)
                                                   │
                  llm_client.py (NASA-expert prompt, history, grounded answer)
                                                   │
                  ragas_evaluator.py (relevancy, faithfulness, BLEU, ROUGE, ...)
                                                   │
                  chat.py (Streamlit UI)   /   batch_evaluate.py (test-set report)
```

## Project structure

| File | Purpose |
|------|---------|
| `embedding_pipeline.py` | Chunks the `.txt` files, embeds them with OpenAI and stores them in ChromaDB (CLI) |
| `rag_client.py` | Backend discovery, semantic retrieval with mission filtering, deduplication, context formatting |
| `llm_client.py` | System prompt, conversation-history management, OpenAI chat call |
| `ragas_evaluator.py` | RAGAS evaluation of a (question, contexts, answer) triple |
| `batch_evaluate.py` | Runs the whole pipeline over the test set; prints results per question and aggregates |
| `chat.py` | Streamlit chat app with live evaluation scores |
| `evaluation_dataset.txt` | 8 test questions with expected answers (overview, emergency, disaster analysis, crew, technical, timeline, grounding) |
| `data_text/` | Source documents: `apollo11/`, `apollo13/`, `challenger/` |

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key-here"
```

**Udacity / Vocareum keys** (starting with `voc-`) work as is. When such a key is used and
`OPENAI_BASE_URL` is not set, every component points the OpenAI clients at
`https://openai.vocareum.com/v1` automatically. With a regular OpenAI key nothing changes.

### RAGAS 0.4.3: VertexAI import fix (required)

RAGAS 0.4.3 imports `ChatVertexAI` and `VertexAI` from old `langchain-community` paths, so
`import ragas` fails with `ImportError: cannot import name 'ChatVertexAI'`. Patch the
installed package once:

```bash
RAGAS_BASE=$(python -c "import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec('ragas').origin), 'llms', 'base.py'))")
sed -i 's/from langchain_community.chat_models.vertexai import ChatVertexAI/from langchain_google_vertexai import ChatVertexAI/' "$RAGAS_BASE"
sed -i 's/from langchain_community.llms import VertexAI/from langchain_google_vertexai import VertexAI/' "$RAGAS_BASE"
python -W ignore -c "import ragas; print('RAGAS:', ragas.__version__)"   # -> RAGAS: 0.4.3
```

(On macOS use `sed -i ''`. Re-apply after reinstalling RAGAS.) See `fix_ragas.png`.

## 1. Build the vector database

```bash
python embedding_pipeline.py --data-path ./data_text
```

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--chunk-size` | `1000` | Maximum characters per chunk (never exceeded) |
| `--chunk-overlap` | `200` | Characters shared by consecutive chunks (must be < chunk size) |
| `--chroma-dir` | `./chroma_db_openai` | ChromaDB persist directory |
| `--collection-name` | `nasa_space_missions_text` | Collection name |
| `--embedding-model` | `text-embedding-3-small` | OpenAI embedding model |
| `--batch-size` | `50` | Chunks embedded per OpenAI request |
| `--update-mode` | `skip` | `skip` existing chunk IDs, `update` them (re-embed and remove stale chunks), or `replace` (delete a file's chunks and re-add) |
| `--stats-only` | | Print collection size and per-mission / category / source counts (no API key needed) |
| `--test-query "..."` | | Run a sample query after processing |
| `--delete-source PATTERN` | | Delete chunks whose source matches a pattern |

The API key comes from `--openai-key` or `OPENAI_API_KEY`. With the defaults, the 12
documents produce **9,340 chunks**. Embedding them with `text-embedding-3-small` costs well
under $0.50.

```bash
python embedding_pipeline.py --stats-only
```

**Chunking strategy.** Whitespace is normalized first. Each chunk is at most `chunk_size`
characters. Its end moves back to the nearest paragraph, sentence, line or word boundary,
but never before the chunk's halfway point. The next chunk starts exactly `chunk_overlap`
characters before the previous one ended, so the overlap is identical between every pair
of chunks.

**Metadata per chunk:** `source`, `file_path`, `mission` (`apollo_11` / `apollo_13` /
`challenger`), `data_type`, `document_category`, `chunk_index`, `total_chunks`,
`chunk_start`, `chunk_end`, `chunk_length`, plus the chunking configuration. IDs are stable
(`<mission>_<source>_chunk_0001`), which is what makes skip and update work.

## 2. Chat

```bash
streamlit run chat.py
```

In the sidebar you choose the collection, model (`gpt-4.1-mini`, `gpt-4.1`, `gpt-4o`),
**mission focus** (a metadata filter), top-k, and whether RAGAS evaluation is on. Each
answer shows its numbered sources and a score line, and the sidebar shows the latest scores.

**How retrieval works.** The collection stores an OpenAI embedding function, so the question
is embedded with the same model as the chunks. ChromaDB returns the nearest chunks by
cosine distance.

With "All missions" selected, a question that names exactly one mission (for example
"What problems did Apollo 13 encounter?") is filtered to that mission automatically. Once a
filter is active, the mission's name is removed from the text that gets embedded, as long
as at least 4 words remain. Almost every chunk repeats its mission's name, so leaving it in
pulled generic status-report chunks to the top. Removing it brings the narrative chunk
describing the 55:58 GET oxygen tank failure to rank 1. The sources panel shows exactly
which text and mission were searched. Twice top-k candidates are fetched, then sorted by distance and
deduplicated, and the best top-k are kept. The context lists each chunk under a
`[Source N] Mission | Document | Category | Chunk | Similarity` header with separators.

**Prompting.** The system prompt sets the model up as a NASA mission-operations expert.
The model must base its answer on the context, cite `[Source N]`, say when the context is
insufficient, and label any general knowledge it adds. The last 10 user/assistant turns
(role + content only) are sent as history. The retrieved context is attached only to the
current question.

## 3. Evaluation

**Real time (every chat answer):** `Response Relevancy` and `Faithfulness`.

**With a reference answer (batch):** also `BLEU` and `ROUGE-L`. With reference contexts,
`Non-LLM Context Precision` is added as well.

```python
from ragas_evaluator import evaluate_response_quality
evaluate_response_quality("What problems did Apollo 13 encounter?", answer, contexts)
# e.g. {'response_relevancy': 0.91, 'faithfulness': 0.86}
```

Empty or malformed input returns `{"error": "<clear message>"}` instead of raising. If one
metric fails, it is reported as `None` with the reason under `metric_errors`, and the other
metrics are still returned.

**Batch evaluation over the test set:**

```bash
python batch_evaluate.py --dataset evaluation_dataset.txt --top-k 3
```

The script prints each question's sources, answer and scores, then a table with mean, min,
max and std for every metric. Detailed results go to `evaluation_results.json`. A JSON
dataset (`[{"question", "expected", "mission", "category"}]`) also works.

### Results (real run, `gpt-4.1-mini`, top-k 5, 9,340 chunks)

Full per-question output, including answers, sources and scores, is in
[`evaluation_results.json`](evaluation_results.json).

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Response Relevancy | 0.785 | 0.000 | 1.000 |
| Faithfulness | 0.841 | 0.500 | 1.000 |
| BLEU | 0.070 | 0.024 | 0.217 |
| ROUGE-L | 0.226 | 0.127 | 0.338 |

How to read these numbers:

- **Q8 (Gemini 4, outside the archive) works as intended.** The model says the documents
  don't cover it and labels McDivitt as general knowledge. RAGAS scores that kind of
  "I don't know" answer as noncommittal, so its relevancy is 0, which lowers the mean.
- **BLEU and ROUGE are low by nature.** They count word overlap with the reference answer,
  and free-form answers word things differently even when the facts match.
- **The data limits some answers.** Most of the archive is raw transcript, so questions that
  need a narrative summary, such as the Q7 timeline, get partial answers.
- **The scores vary between runs.** The LLM-judged metrics changed by a few hundredths
  between identical runs. Before the query cleanup above, Q2 got a vague "oxygen flow"
  answer. After it, Q2 names oxygen tank no. 2 failing at about 55:58 GET, while the
  aggregates stayed within that noise (faithfulness 0.89 vs 0.84, relevancy 0.79 vs 0.79).

![Chat screenshot](docs/chat_screenshot.png)

RAGAS telemetry is disabled by default (`RAGAS_DO_NOT_TRACK=true`).

## Troubleshooting

- **"No ChromaDB backends found"**: run the embedding pipeline from the project directory first.
- **`The CHROMA_OPENAI_API_KEY environment variable is not set`**: set `OPENAI_API_KEY`.
  The app and scripts copy it to `CHROMA_OPENAI_API_KEY` for you.
- **`ImportError: cannot import name 'ChatVertexAI'`**: apply the RAGAS fix above.
- **`sacrebleu` / `rouge_score` missing**: `pip install -r requirements.txt`. BLEU/ROUGE are
  only computed when a reference answer exists.
