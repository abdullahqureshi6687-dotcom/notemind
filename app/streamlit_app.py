"""NoteMind Streamlit app.

Run from the project root:  streamlit run app/streamlit_app.py

Layout:
* Sidebar - upload PDFs (or load the demo notes), settings, document list.
* Chat tab - questions answered from the notes, with expandable sources.
* Quiz tab - multiple-choice quiz from one PDF or a page range.
* How it works tab - architecture summary and live index statistics.

State: each browser session gets its own RAGPipeline in st.session_state,
so users never see each other's documents. The embedding model is loaded
once per server process with st.cache_resource and shared by all sessions.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from notemind.config import (
    API_KEY_ENV_VARS,
    VALID_PROVIDERS,
    AppConfig,
    ConfigError,
    EmbeddingConfig,
    get_api_key,
    load_config,
    setup_logging,
)
from notemind.embedder import Embedder
from notemind.llm_client import (
    LLMAuthError,
    LLMBadRequestError,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from notemind.models import Answer
from notemind.pdf_loader import PDFLoadError
from notemind.pipeline import RAGPipeline
from notemind.quiz import Quiz, QuizGenerationError

st.set_page_config(page_title="NoteMind", page_icon="📚", layout="wide")

MODE_LABELS: dict[str, str] = {
    "hybrid": "Hybrid (embeddings + BM25, RRF)",
    "faiss": "Embeddings (FAISS)",
    "heap": "Embeddings (from-scratch heap)",
    "bm25": "Keywords (BM25)",
}
PROVIDER_LABELS: dict[str, str] = {"gemini": "Google Gemini", "groq": "Groq", "anthropic": "Anthropic Claude"}


# --------------------------------------------------------------------------- cached resources


@st.cache_resource(show_spinner=False)
def get_base_config() -> AppConfig:
    """Load config.yaml once per server process."""
    config = load_config()
    setup_logging(config.log_level)
    return config


@st.cache_resource(show_spinner="Loading the embedding model (the first run downloads about 90 MB)...")
def get_embedder(model_name: str, device: str, batch_size: int) -> Embedder:
    """One embedding model per server process, shared by every session."""
    embedder = Embedder(EmbeddingConfig(model_name=model_name, batch_size=batch_size, device=device))
    _ = embedder.model  # load now, not on the first upload
    return embedder


# --------------------------------------------------------------------------- helpers


def escape_md(text: str) -> str:
    """Stop '$' in notes or answers being rendered as LaTeX math."""
    return text.replace("$", "\\$")


def friendly_error(exc: Exception) -> str:
    """Turn an exception into a message a user can act on."""
    if isinstance(exc, LLMConfigError):
        return f"🔑 {exc}"
    if isinstance(exc, LLMAuthError):
        return "🔑 The API key was rejected. Check the key in your .env file (or the Space secret)."
    if isinstance(exc, LLMRateLimitError):
        return "⏳ The AI provider's rate limit was reached. Wait a minute and try again."
    if isinstance(exc, LLMBadRequestError):
        return f"⚙️ The provider rejected the request. Check the model name in config.yaml. Details: {exc}"
    if isinstance(exc, LLMUnavailableError):
        return "🌐 The AI service is busy or unreachable right now. Please try again shortly."
    return f"⚠️ {exc}"


def init_state(base: AppConfig) -> None:
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("uploaded", {})      # file name -> sha256, for files from the uploader
    ss.setdefault("load_errors", {})   # file name -> error message
    ss.setdefault("uploader_key", 0)   # bump to reset the uploader widget
    ss.setdefault("quiz", None)
    ss.setdefault("quiz_id", 0)
    ss.setdefault("quiz_submitted", False)
    if "pipeline" not in ss:
        embedder = get_embedder(
            base.embedding.model_name, base.embedding.device, base.embedding.batch_size
        )
        ss.pipeline = RAGPipeline(base, embedder=embedder)


def reset_documents() -> None:
    ss = st.session_state
    ss.pipeline.clear()
    ss.uploaded = {}
    ss.load_errors = {}
    ss.uploader_key += 1
    ss.messages = []
    ss.quiz = None
    ss.quiz_submitted = False


# --------------------------------------------------------------------------- sidebar


def sidebar(base: AppConfig) -> AppConfig:
    """Render the sidebar and return the config chosen by the user."""
    ss = st.session_state
    pipeline: RAGPipeline = ss.pipeline

    with st.sidebar:
        st.title("📚 NoteMind")
        st.caption("Chat with your university notes. Answers come only from your PDFs, with page citations.")

        files = st.file_uploader(
            "Upload lecture PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{ss.uploader_key}",
            help="PDFs with selectable text. Scanned/image-only PDFs can't be read (no OCR).",
        )

        demo_pdfs = sorted(base.paths.demo_dir.glob("*.pdf")) if base.paths.demo_dir.is_dir() else []
        if demo_pdfs and st.button("✨ Try demo notes", width="stretch"):
            for pdf in demo_pdfs:
                try:
                    pipeline.add_pdf_path(pdf)
                except (OSError, PDFLoadError) as exc:
                    ss.load_errors[pdf.name] = str(exc)

        with st.expander("⚙️ Settings"):
            chunk_size = st.slider(
                "Chunk size (characters)", 200, 1000, base.chunking.chunk_size, 50,
                help="all-MiniLM-L6-v2 only reads ~256 tokens (~1,000 characters) of each chunk.",
            )
            max_overlap = min(300, chunk_size - 50)
            chunk_overlap = st.slider(
                "Chunk overlap", 0, max_overlap, min(base.chunking.chunk_overlap, max_overlap), 25
            )
            modes = list(MODE_LABELS)
            mode = st.selectbox(
                "Search mode", modes, index=modes.index(base.retrieval.mode),
                format_func=MODE_LABELS.__getitem__,
            )
            top_k = st.slider("Chunks sent to the LLM (top-k)", 1, 10, base.retrieval.top_k)
            min_similarity = st.slider(
                "Refusal threshold (min cosine similarity)", 0.0, 0.6,
                float(base.retrieval.min_similarity), 0.01,
                help="If no chunk is at least this similar, NoteMind answers "
                "'not in your notes' without calling the LLM.",
            )
            providers = list(VALID_PROVIDERS)
            provider = st.selectbox(
                "LLM provider", providers, index=providers.index(base.generation.provider),
                format_func=PROVIDER_LABELS.__getitem__,
            )

        try:
            config = base.with_overrides(
                chunking={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
                retrieval={
                    "mode": mode,
                    "top_k": top_k,
                    "min_similarity": min_similarity,
                    "candidate_k": max(base.retrieval.candidate_k, top_k),
                },
                generation={"provider": provider},
            )
        except ConfigError as exc:
            st.error(f"Invalid settings: {exc}")
            config = base

        if get_api_key(config.generation.provider) is None:
            env_vars = " or ".join(API_KEY_ENV_VARS[config.generation.provider])
            st.warning(
                f"No API key for {PROVIDER_LABELS[config.generation.provider]}. "
                f"Set {env_vars} in .env (local) or as a Space secret. "
                "Retrieval still works; answers and quizzes need the key."
            )

    pipeline.update_config(config)
    sync_uploads(files or [])

    if pipeline.needs_indexing:
        with st.spinner("Indexing your notes..."):
            stats = pipeline.build_index()
        if stats:
            source = "loaded from cache" if stats.from_cache else f"built in {stats.seconds:.1f}s"
            st.toast(f"Indexed {stats.num_chunks} chunks ({source})", icon="✅")

    with st.sidebar:
        documents_panel(pipeline)
    return config


def sync_uploads(files: list) -> None:
    """Make the pipeline's uploaded documents match the uploader widget."""
    ss = st.session_state
    pipeline: RAGPipeline = ss.pipeline
    current = {f.name: f.getvalue() for f in files}

    for name in list(ss.uploaded):
        if name not in current:  # removed with the uploader's "x"
            pipeline.remove_document(name)
            del ss.uploaded[name]
            ss.load_errors.pop(name, None)

    for name, data in current.items():
        digest = hashlib.sha256(data).hexdigest()
        if ss.uploaded.get(name) == digest:
            continue
        ss.uploaded[name] = digest
        try:
            pipeline.add_pdf_bytes(data, name)
            ss.load_errors.pop(name, None)
        except PDFLoadError as exc:
            ss.load_errors[name] = str(exc)


def documents_panel(pipeline: RAGPipeline) -> None:
    ss = st.session_state
    docs = pipeline.documents
    if docs or ss.load_errors:
        st.divider()
        st.subheader("Your documents")
    for doc in docs:
        st.markdown(f"**{doc.source}**  \n{doc.text_pages} of {doc.num_pages} pages have text")
        for warning in doc.report.warnings:
            st.warning(warning, icon="⚠️")
    for name, error in ss.load_errors.items():
        st.error(error, icon="🚫")
    if pipeline.is_indexed:
        st.caption(f"Index: {pipeline.num_chunks} chunks")
    if docs:
        st.button("🗑️ Remove all documents", on_click=reset_documents, width="stretch")


# --------------------------------------------------------------------------- chat


def render_answer(message: dict, config: AppConfig) -> None:
    with st.chat_message("assistant"):
        if message.get("error"):
            st.error(message["error"])
            return
        answer: Answer = message["answer"]
        st.markdown(escape_md(answer.text))

        if answer.refusal_reason == "low_similarity" and answer.max_similarity is not None:
            st.caption(
                f"Best match similarity {answer.max_similarity:.2f} is below the threshold "
                f"{config.retrieval.min_similarity:.2f}, so the LLM was not called."
            )
        if answer.invalid_citations:
            st.warning(
                "This answer cites pages that were not among the retrieved sources: "
                + ", ".join(answer.invalid_citations)
            )
        if answer.sources:
            label = "Closest passages found" if answer.refused else "Sources"
            with st.expander(f"{label} ({len(answer.sources)})"):
                for result in answer.sources:
                    st.markdown(
                        f"**{result.rank}. {result.chunk.citation}** · "
                        f"{result.method} score {result.score:.3f}"
                    )
                    excerpt = result.chunk.text[:700] + ("..." if len(result.chunk.text) > 700 else "")
                    st.caption(escape_md(excerpt))


def chat_tab(config: AppConfig) -> None:
    ss = st.session_state
    pipeline: RAGPipeline = ss.pipeline

    if not pipeline.documents:
        st.info("👈 Upload lecture PDFs in the sidebar (or click **Try demo notes**) to get started.")
    elif not pipeline.is_indexed:
        st.warning("None of your documents contain extractable text, so there is nothing to search.")

    for message in ss.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(escape_md(message["content"]))
        else:
            render_answer(message, config)

    if ss.messages and st.button("Clear chat"):
        ss.messages = []
        st.rerun()

    question = st.chat_input("Ask a question about your notes...", disabled=not pipeline.is_indexed)
    if question:
        ss.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Searching your notes..."):
                answer = pipeline.ask(question)
            ss.messages.append({"role": "assistant", "answer": answer})
        except LLMError as exc:
            ss.messages.append({"role": "assistant", "error": friendly_error(exc)})
        st.rerun()


# --------------------------------------------------------------------------- quiz


def quiz_tab(config: AppConfig) -> None:
    ss = st.session_state
    pipeline: RAGPipeline = ss.pipeline
    docs = [d for d in pipeline.documents if d.text_pages]
    if not docs:
        st.info("Upload a PDF with text to generate a quiz.")
        return

    names = [d.source for d in docs]
    source = st.selectbox("Document", names)
    doc = docs[names.index(source)]
    col_from, col_to, col_n = st.columns(3)
    start = col_from.number_input("From page", 1, doc.num_pages, 1, key=f"start_{source}")
    end = col_to.number_input("To page", 1, doc.num_pages, doc.num_pages, key=f"end_{source}")
    num_questions = col_n.slider(
        "Questions", 1, config.quiz.max_num_questions, config.quiz.default_num_questions
    )

    if st.button("📝 Generate quiz", type="primary"):
        if start > end:
            st.error("'From page' must not be after 'To page'.")
        else:
            try:
                with st.spinner("Writing questions from your notes..."):
                    ss.quiz = pipeline.make_quiz(source, int(num_questions), int(start), int(end))
                ss.quiz_id += 1
                ss.quiz_submitted = False
            except (QuizGenerationError, LLMError) as exc:
                st.error(friendly_error(exc) if isinstance(exc, LLMError) else f"⚠️ {exc}")

    if ss.quiz is not None:
        render_quiz(ss.quiz)


def render_quiz(quiz: Quiz) -> None:
    ss = st.session_state
    st.divider()
    pages = f"pages {quiz.pages[0]}-{quiz.pages[-1]}" if quiz.pages else ""
    st.subheader(f"Quiz: {quiz.source} {pages}")
    for warning in quiz.warnings:
        st.warning(warning)

    keys = [f"quiz_{ss.quiz_id}_{i}" for i in range(len(quiz.questions))]
    with st.form(f"quiz_form_{ss.quiz_id}"):
        for number, (question, key) in enumerate(zip(quiz.questions, keys), start=1):
            page = f" *(p.{question.source_page})*" if question.source_page else ""
            st.markdown(f"**Q{number}. {escape_md(question.question)}**{page}")
            st.radio(
                f"Answer for question {number}",
                question.options,
                index=None,
                key=key,
                label_visibility="collapsed",
                format_func=escape_md,
            )
        if st.form_submit_button("Check answers", type="primary"):
            ss.quiz_submitted = True

    if ss.quiz_submitted:
        correct = sum(ss.get(k) == q.correct_option for q, k in zip(quiz.questions, keys))
        st.metric("Score", f"{correct} / {len(quiz.questions)}")
        for number, (question, key) in enumerate(zip(quiz.questions, keys), start=1):
            chosen = ss.get(key)
            explanation = escape_md(question.explanation)
            if chosen == question.correct_option:
                st.success(f"**Q{number}: correct.** {explanation}")
            else:
                picked = f"You chose “{escape_md(chosen)}”. " if chosen else "Not answered. "
                st.error(
                    f"**Q{number}:** {picked}Correct answer: "
                    f"“{escape_md(question.correct_option)}”. {explanation}"
                )


# --------------------------------------------------------------------------- about


def about_tab(config: AppConfig) -> None:
    pipeline: RAGPipeline = st.session_state.pipeline
    st.markdown(
        """
**How NoteMind answers a question**

1. **Ingest** - each PDF is read page by page (pypdf), so every chunk keeps its page number.
2. **Chunk** - page text is split into overlapping chunks that never cross a page boundary.
3. **Embed** - chunks are embedded with `all-MiniLM-L6-v2` (384-d, runs on CPU).
4. **Retrieve** - dense search (FAISS or a from-scratch heap top-k), BM25 keyword search,
   or both fused with Reciprocal Rank Fusion.
5. **Refuse or generate** - if no chunk is similar enough, NoteMind says the answer is not in
   your notes *without* calling the LLM. Otherwise the LLM answers only from the retrieved
   chunks and must cite `[file, p.X]`. Citations are checked against what was retrieved.
"""
    )
    stats = pipeline.last_index_stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(pipeline.documents))
    col2.metric("Chunks indexed", pipeline.num_chunks)
    col3.metric("Last index build", f"{stats.seconds:.2f}s" if stats else "-",
                "from cache" if stats and stats.from_cache else None)
    st.caption(
        f"Search: {MODE_LABELS[config.retrieval.mode]} · top-k {config.retrieval.top_k} · "
        f"chunk {config.chunking.chunk_size}/{config.chunking.chunk_overlap} · "
        f"LLM: {PROVIDER_LABELS[config.generation.provider]} ({config.generation.model})"
    )


# --------------------------------------------------------------------------- main


def main() -> None:
    try:
        base = get_base_config()
    except ConfigError as exc:
        st.error(f"Configuration error in config.yaml: {exc}")
        st.stop()

    init_state(base)
    config = sidebar(base)

    tab_chat, tab_quiz, tab_about = st.tabs(["💬 Chat", "📝 Quiz", "ℹ️ How it works"])
    with tab_chat:
        chat_tab(config)
    with tab_quiz:
        quiz_tab(config)
    with tab_about:
        about_tab(config)


main()
