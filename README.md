# 📚 Multi-Doc Chat

Chat with several PDFs at once. Upload your books or documents, start a chat with any combination of them, and get answers that **cite the exact passages** they came from, with a one-click view of the source page and the matching text highlighted.

Built with Streamlit, ChromaDB, local FastEmbed embeddings and Groq-hosted LLMs.

## Features

- **Multiple documents per chat.** Pick any set of indexed PDFs for a chat, and change the selection at any time.
- **Covers every document by default.** Unless you name one ("in *Dune*…", "the first book…"), all documents in the chat are searched. Overview questions like *"tell me about all three documents"* sample evenly from each one, so no book gets left out.
- **Cited answers with a source viewer.** Each answer is backed by numbered passages. Open *Show sources* to read the excerpts and view the original PDF page with the passage highlighted.
- **Answer check.** Every answer sentence is compared against the passage it cites using local embeddings. Weakly supported sentences are flagged.
- **Spoiler-safe mode.** Tell the app how far you've read in each book (by chapter) and answers will only use text up to that point.
- **Smart retrieval.** Follow-up questions are rewritten into standalone queries, chapter requests are detected ("summarize Chapter 3"), and irrelevant documents are filtered out for specific questions.
- **Persistent chats.** Conversations are saved locally as JSON and can be reopened or deleted from the sidebar.

## How it works

```
PDF ──► ingest.py ──► chunks + metadata (page, chapter, passage #) ──► ChromaDB (one store per PDF)
                                                                           │
question ──► rag.py ──► pick documents ──► vector search ──► Groq LLM ──► cited answer
                                                                           │
                                       sources + answer check ◄────────────┘
```

1. **Ingest** (`ingest.py`): loads the PDF (PyMuPDF), tags each page with the current chapter, splits text into ~500-character chunks, embeds them with `BAAI/bge-small-en-v1.5` (runs locally via FastEmbed) and saves them in a per-document Chroma store.
2. **Retrieve** (`rag.py`): decides which documents to search, runs a vector search, and keeps a fair share of passages from each document when you ask about several.
3. **Answer** (`rag.py`): sends the numbered passages to a Groq-hosted model, which answers and cites passages as `[1]`, `[2]`, …
4. **Verify** (`rag.py`): builds the source list from the citations and scores each sentence against the passages it cites.

## Project structure

| File | Purpose |
| --- | --- |
| `chatapp.py` | Streamlit UI: uploads, chat list, spoiler-safe mode, sources popup |
| `rag.py` | Retrieval, document routing, prompting, citations, answer check |
| `ingest.py` | PDF loading, chapter detection, chunking, embedding, indexing |
| `config.py` | Shared embedding model setup |
| `chat_store.py` | Saves and loads chats as JSON files |
| `viewer.py` | Renders a PDF page with the cited passage highlighted |
| `docs/` | Uploaded PDFs (created on first run) |
| `vectorstores/` | One Chroma store per PDF (created on first run) |
| `chat_history/` | Saved chats (created on first run) |
| `model_cache/` | Downloaded embedding model (created on first run) |

## Getting started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)

### Install

```bash
git clone https://github.com/Ashlesha95/multi-doc-chat.git
cd multi-doc-chat

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_key_here

# optional: override the default models
# GROQ_MAIN_MODEL=openai/gpt-oss-120b
# GROQ_FAST_MODEL=openai/gpt-oss-20b
```

### Run

```bash
streamlit run chatapp.py
```

The first run downloads the embedding model into `model_cache/`.

## Usage

1. Open **➕ Upload PDFs** in the sidebar, drop in your files and click **Process**. Indexing time depends on the size of the PDF.
2. Under **✏️ New chat**, select one or more documents and click **Start chat**.
3. Ask away. Some examples:
   - *"Tell me about all three documents."*
   - *"Compare how the first and second book handle the ending."*
   - *"Summarize Chapter 4 of Dune."*
   - *"What do the books say about trust?"*
4. Click **📎 Show sources** under an answer to see the passages, the page images and the answer check.
5. Turn on **🛡️ Spoiler-safe mode** and set how far you've read in each book to keep answers spoiler-free. This works for books where chapters are detected (headings like "Chapter 1", "Prologue", "Epilogue").

## Configuration

Tunable constants live at the top of `rag.py` and `ingest.py`:

| Setting | File | Default | What it does |
| --- | --- | --- | --- |
| `GROQ_MAIN_MODEL` | env / `rag.py` | `openai/gpt-oss-120b` | Model that writes answers |
| `GROQ_FAST_MODEL` | env / `rag.py` | `openai/gpt-oss-20b` | Model that rewrites follow-up questions |
| `TOTAL_CHUNKS` | `rag.py` | `12` | Passages sent to the model per question |
| `OVERVIEW_PER_DOC` | `rag.py` | `6` | Passages sampled per document for "tell me about all the books" |
| `DOC_MARGIN` | `rag.py` | `0.10` | How much worse a document's best match can be before it is dropped for specific questions |
| `GROUNDING_THRESHOLD` | `rag.py` | `0.60` | Below this similarity, an answer sentence is flagged as weakly supported |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `ingest.py` | `500` / `50` | Chunking of the PDF text |

## Troubleshooting

- **A document is missing from the answer.** Check that it is listed under *Documents in this chat*, and that the chat box placeholder shows the right count (e.g. "Ask about 3 document(s)…"). The terminal prints a `Docs in play: [...]` line for each question.
- **Weird citation markers like `【7†L1-L6】`.** Some models use their own citation format. The app converts these to `[7]` automatically (`normalize_citations` in `rag.py`).
- **Spoiler-safe mode says no chapters were detected.** Chapter detection looks for lines starting with "Chapter …", "Prologue" or "Epilogue". PDFs with different headings can't be limited by chapter.
- **No text for a page / empty results.** Scanned PDFs without a text layer need OCR before they can be indexed.
- **Re-uploading a file** replaces its existing index, so you won't get duplicates.

## Tech stack

[Streamlit](https://streamlit.io) · [LangChain](https://www.langchain.com) · [ChromaDB](https://www.trychroma.com) · [FastEmbed](https://github.com/qdrant/fastembed) · [PyMuPDF](https://pymupdf.readthedocs.io) · [Groq](https://groq.com)

## License

Add a license of your choice (for example MIT) before sharing the project publicly.
