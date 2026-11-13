# LlamaIndex RAG service

A two-entry retrieval-augmented generation (RAG) service built with LlamaIndex, FastAPI, Pinecone, and OpenAI. One process is a single shared index with document upload. The other creates a Pinecone index per knowledge base.

This repository is a portfolio snapshot of that service. It is not a hosted product, and it does not ship with credentials.

## What it does

You upload text or a supported file. The service extracts text, splits it into chunks, embeds the chunks, and stores them in Pinecone. A question is embedded the same way, matched against those chunks, and passed to an OpenAI chat model with the retrieved text.

There are two FastAPI apps:

| App | Module | Role |
| --- | --- | --- |
| Hybrid server | `hybrid_main.py` | One Pinecone index (`PINECONE_INDEX_NAME`). Upload, status, and query. |
| Knowledge-base API | `main.py` | Creates a separate Pinecone index per knowledge base, then ingests and chats against it. |

Both apps use the retriever in `hybrid_search.py`.

## Architecture

```
upload / ingest
    -> extract text (LlamaParse, or a local fallback)
    -> chunk
    -> OpenAI (or Hugging Face) embeddings
    -> Pinecone

question
    -> semantic search in Pinecone
    -> BM25 over the in-memory chunk corpus
    -> weighted merge
    -> OpenAI completion using the top chunks
```

### Ingestion

`hybrid_main.py` saves the file under `uploads/` and processes it in a FastAPI `BackgroundTasks` job (`LlamaParseHybridProcessor.process_document_pipeline`).

When `LLAMA_CLOUD_API_KEY` is set, PDF, Office, text, markdown, and image files go through LlamaParse. The processor asks LlamaParse for markdown and, for images or names containing `scan`, uses the OCR parser. Other files, and every file when the LlamaParse key is missing, use a local fallback: LlamaIndex `SimpleDirectoryReader` for PDFs, a direct read for text and markdown, and a placeholder string for images. Image OCR is not performed in the fallback path.

Chunks are stored as LlamaIndex `TextNode`s on the Pinecone index. If LlamaParse reports that formatting was preserved, chunks follow markdown-style sections and are capped around 1,500 characters. Otherwise the text is sliced every 1,000 characters.

`main.py` is synchronous from the client's point of view. `KnowledgeBaseManager.ingest` runs LlamaParse when the key is set, otherwise it keeps the original text. Chunking splits on markdown headers, all-caps headings, or numbered sections, then packs sections toward 1,000 characters. The `overlap` argument on that helper is unused.

### Indexing

The hybrid server opens an existing Pinecone index named by `PINECONE_INDEX_NAME` when the process imports `llamaparse_hybrid_processor.py`.

The knowledge-base API creates a new serverless Pinecone index for each knowledge base (`kb-{id}`, AWS, cosine). The region comes from `PINECONE_ENVIRONMENT`, default `us-east-1`. The dimension comes from the embedding model, or 1536 if the model object does not expose one. Index metadata is written to `metadata.json` in the working directory.

Embeddings default to LlamaIndex `OpenAIEmbedding` (OpenAI's default model for that class, using `OPENAI_API_KEY`). The knowledge-base create call can pass another `embedding_model` name, which is loaded with `HuggingFaceEmbedding`.

### Retrieval

`HybridRetriever` runs two searches:

- Semantic: embed the question and query Pinecone (`similarity_top_k` 8).
- Keyword: an in-memory `BM25Okapi` index from `rank-bm25` (`bm25_top_k` 8).

Scores are min-max normalized per list, then combined as `0.7 * semantic + 0.3 * bm25`. Results below `0.1` are dropped. A small fixed synonym list expands queries that mention financial, performance, increase, decrease, table, or report. This is not a general thesaurus.

The BM25 corpus is the chunks processed in the current process. The hybrid server does not rebuild it from Pinecone after a restart. The knowledge-base API rebuilds BM25 from the nodes in the latest ingest call only.

### LLM

`HybridQueryEngine` takes the top five retrieved chunks, builds a prompt that includes scores and matched keywords, and calls `llm.complete`. On the hybrid server the LLM is the index LLM, or `OpenAI(model="gpt-3.5-turbo", temperature=0.1)` if the index has none. The knowledge-base API uses the same shape with `gpt-4o` as its fallback. If hybrid setup fails, the knowledge-base query path falls back to `index.as_query_engine(similarity_top_k=5, response_mode="compact")`.

## Stack

- Python, FastAPI, Uvicorn, Pydantic, python-dotenv
- LlamaIndex (`llama-index`, Pinecone vector store, OpenAI embeddings and LLMs, LlamaParse reader)
- Pinecone, OpenAI, optional LlamaCloud / LlamaParse
- `rank-bm25` and NumPy for keyword search
- `python-magic` for MIME detection on the hybrid path

`requirements_advanced.txt` also lists Celery, Redis, Whisper, and MoviePy. The application code does not call them. Audio and video extensions appear only in the hybrid processor's time-estimate table. They are not transcribed.

Install the dependencies the code imports:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` does not pin versions.

## How to run

```bash
cp .env.example .env
# fill in .env locally; do not commit it
```

Hybrid server (binds `0.0.0.0:8000`):

```bash
python hybrid_main.py
```

Knowledge-base API (pick a different port if the hybrid server is already on 8000):

```bash
uvicorn main:app --port 8002
```

FastAPI serves interactive docs at `/docs` on whichever port you started.

The hybrid process connects to Pinecone at import. Both processes expect `OPENAI_API_KEY` and `PINECONE_API_KEY` to be present before you serve traffic. The knowledge-base API also needs permission to create and delete Pinecone indexes.

### Hybrid server

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -F "file=@data/test_hybrid_search.md"

curl "http://localhost:8000/status/UPLOAD_ID" \
  -H "Authorization: Bearer $RAG_API_KEY"

curl -X POST "http://localhost:8000/query" \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What changed in revenue?"}'
```

`GET /` and `GET /system/health` are public. `GET /search/analytics` reports corpus counts for the current process and does not require a token. `GET /status/{upload_id}` can include an extracted-text preview, so it requires the token.

### Knowledge-base API

Every `/knowledge_bases` route requires the same bearer token. `GET /` and `GET /health` are public.

```bash
curl -X POST "http://localhost:8002/knowledge_bases" \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "demo"}'

curl -X POST "http://localhost:8002/knowledge_bases/KB_ID/documents" \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["LlamaIndex connects data to an LLM."]}'

curl -X POST "http://localhost:8002/knowledge_bases/KB_ID/chat" \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What does LlamaIndex connect?"}'
```

Creating a knowledge base creates a real Pinecone index. Deleting a knowledge base deletes that index.

## Environment variables

See `.env.example`. Names only:

| Variable | Used for |
| --- | --- |
| `OPENAI_API_KEY` | Embeddings and chat completions |
| `PINECONE_API_KEY` | Pinecone client |
| `PINECONE_INDEX_NAME` | Existing index opened by the hybrid server |
| `PINECONE_ENVIRONMENT` | Region for indexes created by the knowledge-base API. Default `us-east-1` |
| `LLAMA_CLOUD_API_KEY` | Optional LlamaParse. Empty disables it |
| `RAG_API_KEY` | Bearer token for protected routes. Unset means those routes return 503 |
| `CORS_ALLOWED_ORIGINS` | Comma-separated browser origins for the hybrid server. `*` is ignored |
| `MAX_UPLOAD_BYTES` | Upload cap. Default 20971520 (20 MiB) |

## Security

Protected routes check `Authorization: Bearer <RAG_API_KEY>` with a SHA-256 digest compared via `hmac.compare_digest`, so the check does not short-circuit on the first differing character or on the length of the configured key. If `RAG_API_KEY` is missing or blank, those routes return 503 and do not accept traffic.

On the hybrid server, CORS origins are the `CORS_ALLOWED_ORIGINS` list. The wildcard origin is discarded. `allow_credentials` is false, so the server does not combine a wildcard with credentialed CORS.

Uploads must use one of: `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.txt`, `.md`, `.rtf`, `.html`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`. The stored name is a sanitized basename plus the upload id. The original path is not used. Files over the size cap are rejected with 413 before they are written.

`GET /`, health checks, and hybrid `GET /search/analytics` do not require a token. Analytics does not return document text. Do not put this process on the public internet without the API key set and without a private network or an additional gateway in front of it.

## Sample data

`data/wisdom.txt` and `data/test_hybrid_search.md` are synthetic documents for a manual upload. Nothing in the servers loads `data/` at startup. The markdown file is a fictional quarterly report used to exercise hybrid search (numbers plus surrounding prose). It is not a client document.

## Layout

```
hybrid_main.py                    Hybrid FastAPI server
llamaparse_hybrid_processor.py    Upload pipeline for that server
main.py                           Knowledge-base FastAPI server
kb_manager.py                     Per-knowledge-base indexes
hybrid_search.py                  Semantic + BM25 retriever and query engine
models.py                         Pydantic models for the hybrid server
schemas.py                        Pydantic models for the knowledge-base API
api_security.py                   Auth, CORS allowlist, upload limits
tests/test_api_security.py        Unit tests for those checks
```

## Tests

The security checks do not need Pinecone or OpenAI:

```bash
python -m unittest tests.test_api_security
python -m compileall -q .
```

The ingestion and query path is not covered by an automated test. Running it requires live OpenAI and Pinecone credentials.

## Limitations

- Upload status and the BM25 corpus live in memory and are lost on restart.
- Keyword search after a restart does not see chunks that exist only in Pinecone.
- The knowledge-base BM25 index is rebuilt from the current ingest, not from the whole index.
- LlamaParse quality features (OCR, tables, layout) run only when `LLAMA_CLOUD_API_KEY` is set and LlamaParse succeeds.
- `requirements_advanced.txt` lists workers and media libraries the Python modules do not use.
- Dependency versions are unpinned.
