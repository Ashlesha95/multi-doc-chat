import os
import re
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from config import get_embeddings

load_dotenv()

# ── Settings ──────────────────────────────────────────────
MAIN_MODEL = os.getenv("GROQ_MAIN_MODEL", "openai/gpt-oss-120b")
FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
TOTAL_CHUNKS = 12
MAX_SUMMARY_CHUNKS = 30
HISTORY_MESSAGES = 6
MAX_MSG_CHARS = 600
EXCERPT_CHARS = 350
GROUNDING_THRESHOLD = 0.60   # below this, a sentence is flagged as weakly supported
DOC_MARGIN = 0.10            # a book is dropped if its best match is this much worse
# than the best book (lower = stricter, higher = looser)
UNCITED_SHOWN = 3            # passages shown when the answer cites nothing

CHAPTER_RE = re.compile(r"\b(prologue|epilogue|chapter\s+[\w-]+)", re.IGNORECASE)
SUMMARY_WORDS = ("summary", "summarize", "summarise", "what happens",
                 "overview", "recap", "tell me about", "describe")
FOLLOWUP_RE = re.compile(
    r"\b(it|its|he|him|his|she|her|hers|they|them|their|that|this|those|these|"
    r"there|then|again|more|also|same|former|latter)\b",
    re.IGNORECASE,
)
COMPARE_RE = re.compile(
    r"\b(compare|comparison|versus|vs|difference|differences|differ|similar|"
    r"similarities|both|between|contrast)\b",
    re.IGNORECASE,
)
GENERIC_RE = re.compile(
    r"\b(books?|documents?|docs?|pdfs?|files?|novels?|texts?)\b", re.IGNORECASE
)
ORDINAL_RE = re.compile(
    r"\b(first|1st|second|2nd|third|3rd)\s+(?:book|document|doc|pdf|file|novel)\b",
    re.IGNORECASE,
)
ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


# ── Cached resources ──────────────────────────────────────
@lru_cache(maxsize=1)
def get_llm():
    return ChatGroq(model=MAIN_MODEL, api_key=os.getenv("GROQ_API_KEY"))


@lru_cache(maxsize=1)
def get_fast_llm():
    return ChatGroq(model=FAST_MODEL, api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0, max_tokens=400)


@lru_cache(maxsize=16)
def get_vectorstore(path: str):
    return Chroma(persist_directory=path, embedding_function=get_embeddings())


@lru_cache(maxsize=32)
def get_chapters(path: str) -> list:
    """Chapters of one document in reading order:
    [{"name": "Chapter One", "first": 3, "last": 17}, ...]  (0-based page indexes)."""
    try:
        data = get_vectorstore(path).get(include=["metadatas"])
    except Exception as e:
        print(f"Chapter list error: {e}")
        return []
    spans, max_page = {}, 0
    for m in data.get("metadatas") or []:
        p, ch = m.get("page"), m.get("chapter")
        if not isinstance(p, int):
            continue
        max_page = max(max_page, p)
        if ch and ch != "Unknown":
            spans[ch] = min(spans.get(ch, p), p)
    ordered = sorted(spans.items(), key=lambda kv: kv[1])
    out = []
    for i, (name, first) in enumerate(ordered):
        last = ordered[i + 1][1] - 1 if i + 1 < len(ordered) else max_page
        out.append({"name": name, "first": first, "last": max(last, first)})
    return out


# ── Helpers (no LLM calls) ────────────────────────────────
def _format_history(chat_history: list) -> str:
    lines = []
    for msg in chat_history[-HISTORY_MESSAGES:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = CITE_RE.sub("", msg["content"]).strip()
        if len(content) > MAX_MSG_CHARS:
            content = content[:MAX_MSG_CHARS] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _needs_rewrite(question: str, chat_history: list) -> bool:
    if not chat_history:
        return False
    return len(question.split()) < 8 or bool(FOLLOWUP_RE.search(question))


def _rewrite_query(question: str, chat_history: list) -> str:
    prompt = (
        "Rewrite the user's last question as a standalone search query, "
        "using the conversation to resolve pronouns and vague references. "
        "Output ONLY the query, nothing else.\n\n"
        f"Conversation:\n{_format_history(chat_history[-4:])}\n\n"
        f"Last question: {question}\n\nStandalone query:"
    )
    try:
        rewritten = get_fast_llm().invoke(prompt).content.strip().strip('"')
        return rewritten or question
    except Exception as e:
        print(f"Rewrite failed, using original question: {e}")
        return question


def _strip_generic(query: str) -> str:
    """Remove 'book/document/pdf' from the search text so it doesn't skew retrieval."""
    cleaned = " ".join(GENERIC_RE.sub(" ", query).split())
    return cleaned if len(cleaned.split()) >= 3 else query


def _target_docs(question: str, doc_paths: dict) -> dict:
    """If the user names a document (or says 'first/second book'), search only that one."""
    if len(doc_paths) <= 1:
        return doc_paths
    q = question.lower().replace("_", " ").replace("-", " ")
    mentioned = {}
    for name, path in doc_paths.items():
        stem = os.path.splitext(name)[0].lower().replace("_", " ").replace("-", " ")
        if len(stem) >= 3 and stem in q:
            mentioned[name] = path
    if mentioned:
        return mentioned

    names = list(doc_paths.keys())
    picks = {ORDINALS[m.group(1).lower()] for m in ORDINAL_RE.finditer(question)}
    chosen = {names[i]: doc_paths[names[i]] for i in picks if i < len(names)}
    return chosen or doc_paths


def _detect_chapter(question: str):
    m = CHAPTER_RE.search(question)
    return m.group(1).strip().title() if m else None


def _build_filter(chapter, max_page):
    conds = []
    if chapter:
        conds.append({"chapter": chapter})
    if max_page is not None:
        conds.append({"page": {"$lte": max_page}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _chapter_locked(doc_paths: dict, chapter: str, limits: dict) -> bool:
    """True if the asked chapter exists but is beyond the reader's progress everywhere."""
    found = False
    for name, path in doc_paths.items():
        lim = limits.get(name)
        for ch in get_chapters(path):
            if ch["name"].lower() == chapter.lower():
                found = True
                if lim is None or ch["first"] <= lim:
                    return False
    return found


def _make_entry(doc_name: str, meta: dict, text: str) -> dict:
    meta = meta or {}
    page = meta.get("page")
    return {
        "doc": doc_name,
        "chapter": meta.get("chapter", "Unknown"),
        "page": page + 1 if isinstance(page, int) else None,
        "para": meta.get("para"),
        "text": text,
    }


def _header(n: int, c: dict) -> str:
    parts = [c["doc"], c["chapter"]]
    if c["page"]:
        parts.append(f"p.{c['page']}")
    if c["para"]:
        parts.append(f"passage {c['para']}")
    return f"[{n}] " + " | ".join(parts)


def _query_store(path: str, embedding: list, k: int, where):
    """Vector search that also returns distances (lower = closer match)."""
    col = get_vectorstore(path)._collection
    res = col.query(
        query_embeddings=[embedding], n_results=k, where=where,
        include=["documents", "metadatas", "distances"],
    )
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def _search_documents(doc_paths: dict, query: str, chapter=None,
                      limits=None, route: bool = True) -> list:
    """Search all selected documents, then drop documents that don't match well."""
    limits = limits or {}
    embedding = get_embeddings().embed_query(query)   # embedded once

    def run(chapter_filter):
        hits = []   # (distance, doc_name, text, metadata)
        for doc_name, path in doc_paths.items():
            try:
                rows = _query_store(
                    path, embedding, TOTAL_CHUNKS,
                    _build_filter(chapter_filter, limits.get(doc_name)),
                )
            except Exception as e:
                print(f"Search error in {doc_name}: {e}")
                continue
            for text, meta, dist in rows:
                hits.append((dist, doc_name, text, meta))
        return hits

    hits = run(chapter)
    if not hits and chapter:
        hits = run(None)
    if not hits:
        return []

    # document routing: keep only documents whose best match is close to the overall best
    if route and len(doc_paths) > 1:
        best = {}
        for dist, name, _, _ in hits:
            best[name] = min(best.get(name, 99.0), dist)
        top = min(best.values())
        keep = {n for n, b in best.items() if b <= top + DOC_MARGIN}
        print(f"Doc routing (best distance per doc): "
              f"{ {n: round(b, 3) for n, b in best.items()} } -> keeping {sorted(keep)}")
        hits = [h for h in hits if h[1] in keep]

    hits.sort(key=lambda h: h[0])
    results, seen = [], set()
    for dist, name, text, meta in hits:
        key = (name, text)
        if key in seen:
            continue
        seen.add(key)
        results.append(_make_entry(name, meta, text))
        if len(results) >= TOTAL_CHUNKS:
            break
    return results


def _get_chapter_chunks(doc_paths: dict, chapter: str, limits=None) -> list:
    limits = limits or {}
    per_doc_cap = max(5, MAX_SUMMARY_CHUNKS // len(doc_paths))
    results = []
    for doc_name, path in doc_paths.items():
        try:
            data = get_vectorstore(path).get(
                where=_build_filter(chapter, limits.get(doc_name)))
        except Exception as e:
            print(f"Chapter fetch error in {doc_name}: {e}")
            continue
        entries = [
            _make_entry(doc_name, m, t)
            for t, m in zip(data.get("documents") or [], data.get("metadatas") or [])
        ]
        entries.sort(key=lambda e: (e["page"] or 0, e["para"] or 0))
        if len(entries) > per_doc_cap:
            step = len(entries) / per_doc_cap
            entries = [entries[int(i * step)] for i in range(per_doc_cap)]
        results.extend(entries)
    return results


def _build_sources(answer: str, chunks: list) -> list:
    """Cited passages only. If nothing was cited, show just the few closest ones."""
    cited = set()
    for m in CITE_RE.finditer(answer):
        for x in re.split(r"\s*,\s*", m.group(1)):
            if x.isdigit() and 1 <= int(x) <= len(chunks):
                cited.add(int(x))

    chosen = sorted(cited) if cited else range(1, min(UNCITED_SHOWN, len(chunks)) + 1)
    sources = []
    for n in chosen:
        c = chunks[n - 1]
        flat = " ".join(c["text"].split())
        sources.append({
            "n": n, "doc": c["doc"], "chapter": c["chapter"],
            "page": c["page"], "para": c["para"],
            "excerpt": flat[:EXCERPT_CHARS] + ("..." if len(flat) > EXCERPT_CHARS else ""),
            "text": flat,
            "cited": bool(cited),
        })
    return sources


def _grounding_check(answer: str, chunks: list) -> list:
    """Compare each answer sentence with the passages it cites (local embeddings only)."""
    if not CITE_RE.search(answer) or not chunks:
        return []
    emb = get_embeddings()

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]
    items = []
    for s in sentences:
        clean = CITE_RE.sub("", s).strip(" -*•")
        if len(clean.split()) >= 5:
            ids = {int(x) for m in CITE_RE.finditer(s)
                   for x in re.split(r"\s*,\s*", m.group(1))
                   if x.isdigit() and 1 <= int(x) <= len(chunks)}
            items.append((clean, ids))
    if not items:
        return []

    def norm(a):
        a = np.array(a, dtype=float)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

    chunk_vecs = norm(emb.embed_documents([c["text"] for c in chunks]))
    sent_vecs = norm(emb.embed_documents([s for s, _ in items]))
    sims = sent_vecs @ chunk_vecs.T

    out = []
    for (sentence, ids), row in zip(items, sims):
        idx = [i - 1 for i in ids] if ids else list(range(len(chunks)))
        score = float(row[idx].max())
        out.append({
            "sentence": sentence,
            "score": round(score, 2),
            "weak": score < GROUNDING_THRESHOLD,
            "cited": bool(ids),
        })
    return out


# ── Main entry point ──────────────────────────────────────
def smart_answer(doc_paths: dict, question: str, chat_history: list = None,
                 limits: dict = None):
    """
    limits: {"Book A.pdf": last_allowed_page_index} for spoiler-safe mode (optional).
    Returns (answer, sources, grounding).
    LLM calls: 1 normally, 2 when a follow-up needs rewriting.
    """
    chat_history = chat_history or []
    limits = {k: v for k, v in (limits or {}).items() if v is not None}
    doc_paths = _target_docs(question, doc_paths)

    search_query = (_rewrite_query(question, chat_history)
                    if _needs_rewrite(question, chat_history) else question)
    search_query = _strip_generic(search_query)
    print(f"Search query: {search_query}")

    chapter = _detect_chapter(search_query) or _detect_chapter(question)
    is_summary = any(w in question.lower() for w in SUMMARY_WORDS)
    is_comparison = bool(COMPARE_RE.search(question))

    if limits and chapter and _chapter_locked(doc_paths, chapter, limits):
        return ("🔒 Spoiler-safe mode is on, and that chapter is beyond where you've "
                "read so far. Move your reading progress forward to unlock it.", [], [])

    if chapter and is_summary:
        chunks = _get_chapter_chunks(doc_paths, chapter, limits) or \
                 _search_documents(doc_paths, search_query, None, limits,
                                   route=not is_comparison)
    else:
        chunks = _search_documents(doc_paths, search_query, chapter, limits,
                                   route=not is_comparison)

    if not chunks:
        return ("I couldn't find anything relevant in the part of the documents "
                "you're allowed to see." if limits else
                "I couldn't find anything relevant in the selected documents.", [], [])

    # only the documents that actually made it into the context are named in the prompt
    doc_names = list(dict.fromkeys(c["doc"] for c in chunks))

    context = "\n\n---\n\n".join(
        f"{_header(i, c)}\n{c['text']}" for i, c in enumerate(chunks, 1)
    )
    history_text = _format_history(chat_history) or "No previous conversation."

    spoiler_rule = (
        "- SPOILER-SAFE MODE: the reader has NOT finished these books. Use ONLY the passages "
        "below. Never reveal, hint at, or add outside knowledge about later events."
        if limits else ""
    )
    summary_rule = ("- This is a request to summarize a chapter: give a coherent "
                    "summary in story order." if chapter and is_summary else "")

    prompt = f"""You are a helpful, conversational assistant helping someone understand
these documents: {", ".join(doc_names)}.
The user may call them books, documents, PDFs or files. These all mean the same thing:
the passages below.
Each passage below starts with a number like [1] followed by its document, chapter and page.

Guidelines:
- Answer naturally, as if you know the material well. Don't say "based on the context".
- After each claim or sentence that uses a passage, cite it with its number, e.g. [2] or [1][3].
  Only cite passages you actually used. Never invent a citation.
- When it matters which document something comes from, mention it (e.g. "In Book A, ...").
- For comparisons, cover each document separately, then compare.
- Never mix up facts between documents.
- If the question is vague, use the conversation history to work out what they mean.
- If the passages don't contain the answer, say so honestly and don't cite anything.
{spoiler_rule}
{summary_rule}

Conversation history:
{history_text}

Passages:
{context}

User: {question}

Answer:"""

    answer = get_llm().invoke(prompt).content
    return answer, _build_sources(answer, chunks), _grounding_check(answer, chunks)