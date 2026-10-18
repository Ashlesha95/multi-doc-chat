import streamlit as st
import os
import re
import shutil

from rag import smart_answer, get_vectorstore, get_chapters, normalize_citations
from ingest import ingest
from viewer import render_highlighted_page
from chat_store import (
    create_new_chat, save_chat, load_chat,
    get_all_chats, delete_chat, generate_title
)

DOCS_DIR = "./docs"
VS_DIR = "./vectorstores"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(VS_DIR, exist_ok=True)

CITE_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
ALL_READ = "(I've read everything)"


def get_vectorstore_path(pdf_name: str) -> str:
    return os.path.join(VS_DIR, os.path.splitext(pdf_name)[0])


def get_indexed_pdfs() -> list:
    return sorted(
        f for f in os.listdir(DOCS_DIR)
        if f.endswith(".pdf") and os.path.exists(get_vectorstore_path(f))
    )


def clean_answer(text: str) -> str:
    """Hide the [1], [2] markers; the sources stay available behind the button.
    Also converts model-native 【7†L1-L6】 markers first, so old saved chats
    that still contain them render cleanly too."""
    return CITE_RE.sub("", normalize_citations(text))


def clear_caches():
    get_vectorstore.cache_clear()
    get_chapters.cache_clear()


@st.cache_data(show_spinner=False)
def page_image(pdf_name: str, page: int, text: str):
    return render_highlighted_page(os.path.join(DOCS_DIR, pdf_name), page, text)


def toggle(key: str):
    st.session_state[key] = not st.session_state.get(key, False)


def render_weak_flag(grounding):
    if grounding and any(g["weak"] for g in grounding):
        st.caption("⚠️ Some parts may be weakly supported. Open the sources to check.")


def render_answer_check(grounding):
    if not grounding:
        return
    weak = [g for g in grounding if g["weak"]]
    st.markdown("**Answer check**")
    if not weak:
        st.caption("✅ Every sentence matches the passage it cites.")
    else:
        st.caption(f"⚠️ {len(weak)} of {len(grounding)} sentences look weakly supported:")
        for g in weak:
            st.markdown(f"- “{g['sentence']}” *(match {g['score']:.2f})*")
    st.divider()


# ── Popups ────────────────────────────────────────────────
@st.dialog("📎 Sources", width="large")
def sources_dialog(chat_id: str, idx: int, sources: list, grounding: list):
    render_answer_check(grounding)
    if not sources[0].get("cited", True):
        st.caption("The answer didn't cite anything directly. "
                   "These are the closest passages that were searched.")
    for i, s in enumerate(sources, 1):
        loc = f"**{i}. {s['doc']}** · {s['chapter']}"
        if s.get("page"):
            loc += f" · page {s['page']}"
        if s.get("para"):
            loc += f" · passage {s['para']}"
        st.markdown(loc)
        st.caption(f"“{s['excerpt']}”")

        if s.get("page") and os.path.exists(os.path.join(DOCS_DIR, s["doc"])):
            pkey = f"pg_{chat_id}_{idx}_{i}"
            pshown = st.session_state.get(pkey, False)
            st.button("🙈 Hide page" if pshown else "🖼️ View page",
                      key=f"btn_{pkey}", on_click=toggle, args=(pkey,))
            if pshown:
                try:
                    img, found = page_image(s["doc"], s["page"], s.get("text", s["excerpt"]))
                    st.image(img, use_container_width=True)
                    if not found:
                        st.caption("Couldn't locate the exact text on the page, "
                                   "so it's shown without a highlight.")
                except Exception as e:
                    st.caption(f"Couldn't render the page: {e}")
        if i < len(sources):
            st.divider()


@st.dialog("⚠️ Turn off spoiler-safe mode?")
def confirm_spoilers_off(chat_id: str):
    st.warning(
        "Without spoiler-safe mode, answers can use **any part of the books**, "
        "so they may reveal plot twists, endings or other events you haven't "
        "reached yet."
    )
    st.caption("Your reading progress is kept, so you can turn the mode back on later.")
    c1, c2 = st.columns(2)
    if c1.button("Keep it on", use_container_width=True):
        st.rerun()
    if c2.button("Turn off anyway", type="primary", use_container_width=True):
        ch = load_chat(chat_id)
        if ch:
            ch["spoiler_safe"] = False
            save_chat(ch)
        st.rerun()


def render_sources_button(chat_id: str, idx: int, sources: list, grounding: list):
    """Sources are always prepared; the popup only opens when the user clicks."""
    if not sources:
        return
    if st.button("📎 Show sources", key=f"btn_src_{chat_id}_{idx}"):
        sources_dialog(chat_id, idx, sources, grounding)


# ── Session state ─────────────────────────────────────────
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

indexed = get_indexed_pdfs()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📚 RAG Chatbot")

    with st.expander("➕ Upload PDFs", expanded=False):
        uploaded_files = st.file_uploader(
            "Drop PDFs here", type="pdf",
            accept_multiple_files=True, label_visibility="collapsed"
        )
        if uploaded_files and st.button("Process", use_container_width=True):
            bar = st.progress(0.0, text="Starting...")
            total = len(uploaded_files)
            for idx, uf in enumerate(uploaded_files):
                file_path = os.path.join(DOCS_DIR, uf.name)
                with open(file_path, "wb") as f:
                    f.write(uf.getbuffer())
                clear_caches()
                ingest(
                    file_path, get_vectorstore_path(uf.name),
                    progress=lambda frac, i=idx, n=uf.name: bar.progress(
                        (i + frac) / total, text=f"Indexing {n}..."),
                )
            clear_caches()
            bar.empty()
            st.success("✅ Ready!")
            st.rerun()

    with st.expander(f"📄 Documents ({len(indexed)})", expanded=False):
        if not indexed:
            st.caption("No documents yet.")
        for pdf in indexed:
            c1, c2 = st.columns([5, 1])
            c1.write(pdf)
            if c2.button("🗑️", key=f"deldoc_{pdf}"):
                clear_caches()
                os.remove(os.path.join(DOCS_DIR, pdf))
                vs = get_vectorstore_path(pdf)
                if os.path.exists(vs):
                    shutil.rmtree(vs)
                st.rerun()

    st.divider()
    st.subheader("✏️ New chat")
    new_chat_docs = st.multiselect("Documents", indexed, key="new_chat_docs")
    if st.button("Start chat", use_container_width=True, disabled=not new_chat_docs):
        chat = create_new_chat(new_chat_docs)
        st.session_state.active_chat_id = chat["id"]
        st.rerun()

    st.divider()
    st.subheader("💬 Chats")
    chats = get_all_chats()
    if not chats:
        st.caption("No chats yet")
    for c in chats:
        is_active = st.session_state.active_chat_id == c["id"]
        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"{'→ ' if is_active else ''}{c['title']}"
            if st.button(label, key=f"chat_{c['id']}", use_container_width=True,
                         help=", ".join(c.get("docs", []))):
                st.session_state.active_chat_id = c["id"]
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{c['id']}"):
                delete_chat(c["id"])
                if is_active:
                    st.session_state.active_chat_id = None
                st.rerun()

# ── Main area ─────────────────────────────────────────────
chat = load_chat(st.session_state.active_chat_id) if st.session_state.active_chat_id else None

if not chat:
    st.title("RAG Chatbot 🤖")
    st.info("👈 Pick one or more documents and start a new chat from the sidebar!")
    st.stop()

st.title(f"💬 {chat['title']}")
st.caption(f"Created {chat.get('created_at', '')}")

current_docs = [d for d in chat["docs"] if d in indexed]
selected = st.multiselect(
    "Documents in this chat", indexed,
    default=current_docs, key=f"docs_{chat['id']}"
)
if set(selected) != set(chat["docs"]):
    chat["docs"] = selected
    save_chat(chat)

# ── Spoiler-safe mode (can be switched at any time) ───────
spoiler_on = bool(chat.get("spoiler_safe"))

if spoiler_on:
    if st.button("🛡️ Spoiler-safe mode: ON · click to turn off",
                 key=f"spoil_off_{chat['id']}"):
        confirm_spoilers_off(chat["id"])  # opens the warning popup
else:
    if st.button("🛡️ Spoiler-safe mode: OFF · click to turn on",
                 key=f"spoil_on_{chat['id']}"):
        chat["spoiler_safe"] = True
        save_chat(chat)
        st.rerun()

if spoiler_on:
    with st.expander("📖 Reading progress", expanded=True):
        progress = chat.setdefault("read_up_to", {})
        progress_changed = False
        for doc in selected:
            chapters = get_chapters(get_vectorstore_path(doc))
            if not chapters:
                st.caption(f"{doc}: no chapters were detected, so it can't be limited.")
                continue
            names = [c["name"] for c in chapters]
            current = progress.get(doc)
            choice = st.selectbox(
                f"{doc}: I've finished up to",
                [ALL_READ] + names,
                index=(names.index(current) + 1) if current in names else 0,
                key=f"read_{chat['id']}_{doc}",
                )
            new_val = None if choice == ALL_READ else choice
            if new_val != current:
                progress[doc] = new_val
                progress_changed = True
        if progress_changed:
            save_chat(chat)
        if not any(progress.get(d) for d in selected):
            st.caption("Pick how far you've read in each book. Until you do, "
                       "nothing is hidden.")
        else:
            st.caption("Answers only use text up to the end of the chapter you pick.")

limits = {}
if spoiler_on:
    for doc in selected:
        name = chat.get("read_up_to", {}).get(doc)
        if name:
            for ch in get_chapters(get_vectorstore_path(doc)):
                if ch["name"] == name:
                    limits[doc] = ch["last"]

# ── History ───────────────────────────────────────────────
for idx, m in enumerate(chat["messages"]):
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            st.markdown(clean_answer(m["content"]))
            render_weak_flag(m.get("grounding"))
            render_sources_button(chat["id"], idx, m.get("sources"), m.get("grounding"))
        else:
            st.markdown(m["content"])

if not selected:
    st.warning("Select at least one document to chat with.")
    st.stop()

# ── Chat input ────────────────────────────────────────────
if prompt := st.chat_input(f"Ask about {len(selected)} document(s)..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources, grounding = smart_answer(
                doc_paths={d: get_vectorstore_path(d) for d in selected},
                question=prompt,
                chat_history=chat["messages"],
                limits=limits,
            )
        st.markdown(clean_answer(answer))

    chat["messages"] += [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer,
         "sources": sources, "grounding": grounding},
    ]
    if chat["title"] == "New Chat":
        chat["title"] = generate_title(prompt)
    save_chat(chat)
    st.rerun()