import os
import re
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import get_embeddings

try:
    from langchain_community.document_loaders import PyMuPDFLoader as PDFLoader
except ImportError:  # falls back to the slow loader if pymupdf isn't installed
    from langchain_community.document_loaders import PyPDFLoader as PDFLoader

CHUNK_SIZE = 500       # raising to 800 gives fewer chunks, so faster indexing, coarser excerpts
CHUNK_OVERLAP = 50
BATCH = 128

CHAPTER_PATTERN = re.compile(
    r'^(prologue|epilogue|chapter\s+[\w-]+)',
    re.IGNORECASE | re.MULTILINE
)


def detect_chapter(text: str):
    match = CHAPTER_PATTERN.search(text.strip())
    return match.group(0).strip().title() if match else None


def ingest(pdf_path: str, vectorstore_dir: str, progress=None):
    """progress: optional callback taking a float from 0 to 1."""
    t0 = time.time()
    doc_name = os.path.basename(pdf_path)

    # 1) load (PyMuPDF is much faster than pypdf)
    docs = PDFLoader(pdf_path).load()
    docs = [d for d in docs if d.page_content.strip()]    # skip blank pages
    t_load = time.time()

    # 2) tag chapters; keep only the metadata we use (smaller, faster store)
    current_chapter = "Unknown"
    for d in docs:
        detected = detect_chapter(d.page_content)
        if detected:
            current_chapter = detected
        d.metadata = {
            "page": d.metadata.get("page"),
            "chapter": current_chapter,
            "doc_name": doc_name,
        }

    # 3) split + number the passages on each page
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    per_page = {}
    for c in chunks:
        p = c.metadata.get("page")
        per_page[p] = per_page.get(p, 0) + 1
        c.metadata["para"] = per_page[p]
    t_split = time.time()

    # 4) start from a clean store so re-uploading doesn't create duplicates
    embeddings = get_embeddings()
    if os.path.exists(vectorstore_dir):
        try:
            Chroma(persist_directory=vectorstore_dir,
                   embedding_function=embeddings).delete_collection()
        except Exception as e:
            print(f"Could not clear old store: {e}")
    store = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)

    # 5) embed + save in batches, reporting progress
    for i in range(0, len(chunks), BATCH):
        store.add_documents(chunks[i:i + BATCH])
        if progress:
            progress(min(i + BATCH, len(chunks)) / len(chunks))
    t_end = time.time()

    print(f"✅ {doc_name}: {len(docs)} pages, {len(chunks)} chunks | "
          f"load {t_load - t0:.1f}s, split {t_split - t_load:.1f}s, "
          f"embed+save {t_end - t_split:.1f}s, total {t_end - t0:.1f}s")