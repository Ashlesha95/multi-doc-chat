<div align="center">

# 📚 Multi-Doc Chat

### Chat with several PDFs at once, with answers you can actually verify.

Upload your books or documents, start a chat with any combination of them, and get answers that **cite the exact passages** they came from, with the original PDF page highlighted.

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge)

[**Features**](#-features) · [**How it works**](#-how-it-works) · [**Quick start**](#-quick-start) · [**Usage**](#-usage) · [**Configuration**](#%EF%B8%8F-configuration) · [**Troubleshooting**](#-troubleshooting)

</div>

<br/>

<!--
  📸 Add a screenshot or GIF here once you have one:
  1. create an `assets/` folder and drop in `demo.png`
  2. uncomment the line below

  <p align="center"><img src="assets/demo.png" alt="Multi-Doc Chat demo" width="85%"/></p>
-->

## ✨ Features

| | Feature | What it does |
| :-: | --- | --- |
| 📖 | **Multi-document chats** | Pick any set of indexed PDFs for a chat and change the selection whenever you like. |
| 🎯 | **Covers every document** | Unless you name one (*"in Dune…"*, *"the first book…"*), **all** documents in the chat are searched. Overview questions like *"tell me about all three documents"* sample evenly from each book, so none gets left out. |
| 📎 | **Cited answers** | Every answer is backed by numbered passages. Open *Show sources* to read the excerpts and see the PDF page with the passage highlighted. |
| ✅ | **Answer check** | Each sentence is compared with the passage it cites using local embeddings. Weakly supported sentences get flagged. |
| 🛡️ | **Spoiler-safe mode** | Say how far you've read in each book (by chapter) and answers only use text up to that point. |
| 🧠 | **Smart retrieval** | Follow-ups are rewritten into standalone queries, chapter requests are detected (*"summarize Chapter 3"*), and unrelated documents are filtered out for specific questions. |
| 💾 | **Persistent chats** | Conversations are saved locally as JSON, so you can reopen or delete them from the sidebar. |
| 🔒 | **Local embeddings** | Documents are embedded on your machine with FastEmbed. Only the retrieved passages are sent to the LLM. |

## 🔍 How it works

```text
  ┌────────────┐      ┌───────────────────────────┐      ┌─────────────────────┐
  │    PDFs    │ ───▶ │  ingest.py                │ ───▶ │  ChromaDB           │
  │            │      │  chunk + tag page/chapter │      │  one store per PDF  │
  └────────────┘      └───────────────────────────┘      └──────────┬──────────┘
                                                                    │
  ┌────────────┐      ┌───────────────────────────┐                 │
  │  Question  │ ───▶ │  rag.py                   │ ◀───────────────┘
  │            │      │  pick docs + retrieve     │
  └────────────┘      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐      ┌─────────────────────┐
                      │  Groq LLM                 │ ───▶ │  Cited answer       │
                      └───────────────────────────┘      │  Sources            │
                                                         │  Answer check       │
                                                         └─────────────────────┘
```

1. **Ingest** (`ingest.py`): loads each PDF with PyMuPDF, tags every page with its chapter, splits the text into ~500-character chunks, embeds them with `BAAI/bge-small-en-v1.5` (locally, via FastEmbed) and saves them in a per-document Chroma store.
2. **Retrieve** (`rag.py`): decides which documents to search, runs a vector search, and keeps a fair share of passages from every document when you ask about several.
3. **Answer** (`rag.py`): sends the numbered passages to a Groq-hosted model, which answers and cites them as `[1]`, `[2]`, …
4. **Verify** (`rag.py`): builds the source list from the citations and scores each sentence against the passages it cites.

## 🚀 Quick start

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

> 💡 The first run downloads the embedding model into `model_cache/`, so it takes a little longer.

## 💬 Usage

1. **Upload.** Open **➕ Upload PDFs** in the sidebar, drop in your files and click **Process**.
2. **Start a chat.** Under **✏️ New chat**, select one or more documents and click **Start chat**.
3. **Ask.** Try things like:

   | Question | What happens |
   | --- | --- |
   | *"Tell me about all three documents."* | Samples passages evenly from every document and covers each one |
   | *"What do the books say about trust?"* | Searches all documents, with a fair share of passages from each |
   | *"Summarize Chapter 4 of Dune."* | Narrows to that document and chapter |
   | *"Compare how the first and second book end."* | Targets those two books and compares them side by side |

4. **Check the sources.** Click **📎 Show sources** under an answer to see the passages, the page images and the answer check.
5. **Avoid spoilers.** Turn on **🛡️ Spoiler-safe mode** and set how far you've read in each book. This works for books where chapters are detected (headings like *Chapter 1*, *Prologue*, *Epilogue*).

## 🗂️ Project structure

```
multi-doc-chat/
├── chatapp.py       # Streamlit UI: uploads, chats, spoiler-safe mode, sources popup
├── rag.py           # retrieval, document routing, prompting, citations, answer check
├── ingest.py        # PDF loading, chapter detection, chunking, embedding, indexing
├── config.py        # shared embedding model setup
├── chat_store.py    # saves and loads chats as JSON
├── viewer.py        # renders a PDF page with the cited passage highlighted
│
├── docs/            # uploaded PDFs                (created on first run)
├── vectorstores/    # one Chroma store per PDF     (created on first run)
├── chat_history/    # saved chats                  (created on first run)
└── model_cache/     # downloaded embedding model   (created on first run)
```

## ⚙️ Configuration

<details>
<summary><b>Tunable settings</b> (click to expand)</summary>

<br/>

These live at the top of `rag.py` and `ingest.py`:

| Setting | File | Default | What it does |
| --- | --- | --- | --- |
| `GROQ_MAIN_MODEL` | env / `rag.py` | `openai/gpt-oss-120b` | Model that writes answers |
| `GROQ_FAST_MODEL` | env / `rag.py` | `openai/gpt-oss-20b` | Model that rewrites follow-up questions |
| `TOTAL_CHUNKS` | `rag.py` | `12` | Passages sent to the model per question |
| `OVERVIEW_PER_DOC` | `rag.py` | `6` | Passages sampled per document for "tell me about all the books" |
| `DOC_MARGIN` | `rag.py` | `0.10` | How much worse a document's best match can be before it is dropped for specific questions |
| `GROUNDING_THRESHOLD` | `rag.py` | `0.60` | Below this similarity, an answer sentence is flagged as weakly supported |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `ingest.py` | `500` / `50` | How the PDF text is chunked |

</details>

## 🩺 Troubleshooting

<details>
<summary><b>A document is missing from the answer</b></summary>

<br/>

Check that it's listed under **Documents in this chat** and that the chat box placeholder shows the right count (for example *"Ask about 3 document(s)…"*). The terminal also prints a `Docs in play: [...]` line for every question, which shows exactly which documents were searched.

</details>

<details>
<summary><b>Odd citation markers like <code>【7†L1-L6】</code></b></summary>

<br/>

Some models use their own citation format instead of `[7]`. The app converts these automatically (`normalize_citations` in `rag.py`). If you still see odd markers, the model may be using a format the regex doesn't cover yet.

</details>

<details>
<summary><b>Spoiler-safe mode says no chapters were detected</b></summary>

<br/>

Chapter detection looks for lines starting with *Chapter …*, *Prologue* or *Epilogue*. PDFs that use different headings can't be limited by chapter.

</details>

<details>
<summary><b>Empty results or missing text</b></summary>

<br/>

Scanned PDFs without a text layer need OCR before they can be indexed. Re-uploading a file replaces its existing index, so you won't get duplicates.

</details>

## 🧰 Tech stack

[Streamlit](https://streamlit.io) · [LangChain](https://www.langchain.com) · [ChromaDB](https://www.trychroma.com) · [FastEmbed](https://github.com/qdrant/fastembed) · [PyMuPDF](https://pymupdf.readthedocs.io) · [Groq](https://groq.com)

## 📄 License

Released under the [MIT License](LICENSE).

<br/>

<div align="center">

⭐ If you find this useful, consider giving the repo a star.

</div>
