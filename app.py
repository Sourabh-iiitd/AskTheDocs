import os

import numpy as np
import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from ask import load_index, build_prompt, generate_with_retry
from ingest import chunk_text

TOP_K = 4
MIN_SCORE = 0.25      # below this, the question is treated as "not in the documents"
CHUNK_SIZE = 800
OVERLAP = 150

st.set_page_config(page_title="AskTheDocs", page_icon="📄")
st.title("📄 AskTheDocs")
st.caption("Ask questions about your documents. Answers come only from the documents, with sources. Demo: UIDAI internship documents.")


@st.cache_resource
def load_everything():
    chunks, embeddings = load_index()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return chunks, embeddings, embed_model, client


chunks, embeddings, embed_model, client = load_everything()

# ---------- session memory ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "up_chunks" not in st.session_state:
    st.session_state.up_chunks = []        # chunks from uploaded files
    st.session_state.up_embeddings = None  # their vectors
    st.session_state.up_names = []         # files successfully added
    st.session_state.up_seen = []          # files already processed (even if failed)


# ---------- helpers ----------
def read_uploaded(file):
    """Return a list of (page_number, text) from an uploaded PDF or txt file."""
    pages = []
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i + 1, text))
    else:
        text = file.read().decode("utf-8", errors="ignore")
        if text.strip():
            pages.append((1, text))
    return pages


def add_file(file):
    """Chunk + embed an uploaded file and add it to this session's memory. Returns number of chunks."""
    new_chunks = []
    for page_no, text in read_uploaded(file):
        for piece in chunk_text(text, CHUNK_SIZE, OVERLAP):
            new_chunks.append({"source": file.name, "page": page_no, "text": piece})
    if not new_chunks:
        return 0
    vecs = embed_model.encode([c["text"] for c in new_chunks], normalize_embeddings=True)
    st.session_state.up_chunks += new_chunks
    prev = st.session_state.up_embeddings
    st.session_state.up_embeddings = vecs if prev is None else np.vstack([prev, vecs])
    st.session_state.up_names.append(file.name)
    return len(new_chunks)


def retrieve_with_scores(question, only_uploaded, k=TOP_K):
    """Pick the pool of chunks to search, then return the top k with similarity scores."""
    pool_chunks, pool_emb = chunks, embeddings
    if st.session_state.up_chunks:
        if only_uploaded:
            pool_chunks, pool_emb = st.session_state.up_chunks, st.session_state.up_embeddings
        else:
            pool_chunks = chunks + st.session_state.up_chunks
            pool_emb = np.vstack([embeddings, st.session_state.up_embeddings])
    q_vec = embed_model.encode([question], normalize_embeddings=True)[0]
    scores = pool_emb @ q_vec
    best = scores.argsort()[::-1][:k]
    return [(pool_chunks[i], float(scores[i])) for i in best]


def show_sources(sources):
    with st.expander("Sources"):
        for c, score in sources:
            st.markdown(f"**{c['source']}** (page {c['page']}) · match: {score:.2f}")
            st.text(c["text"][:400])


# ---------- sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("RAG app: finds the most relevant passages in the documents, then asks an LLM to answer only from them.")

    st.subheader("Upload your document")
    uploads = st.file_uploader("PDF or .txt", type=["pdf", "txt"], accept_multiple_files=True)
    for f in uploads or []:
        if f.name not in st.session_state.up_seen:
            st.session_state.up_seen.append(f.name)
            with st.spinner(f"Reading {f.name}..."):
                n = add_file(f)
            if n == 0:
                st.warning(f"No readable text in {f.name} (is it a scanned PDF?).")
            else:
                st.success(f"Added {f.name} ({n} chunks)")
    st.caption("Uploaded text is sent to the Gemini API to generate answers. Do not upload personal or confidential documents.")

    only_uploaded = False
    if st.session_state.up_names:
        scope = st.radio("Search in", ["All documents", "Only my uploads"])
        only_uploaded = scope == "Only my uploads"

    st.subheader("Documents loaded")
    for name in sorted({c["source"] for c in chunks}):
        st.write("•", name)
    for name in st.session_state.up_names:
        st.write("•", name, "(uploaded)")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if st.session_state.up_names and st.button("Remove uploaded documents"):
        st.session_state.up_chunks = []
        st.session_state.up_embeddings = None
        st.session_state.up_names = []
        st.session_state.up_seen = []
        st.rerun()

# ---------- chat ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

question = st.chat_input("Ask about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            results = retrieve_with_scores(question, only_uploaded)
            best_score = results[0][1]

            if best_score < MIN_SCORE:
                answer = "I couldn't find this in the documents. Try rephrasing, or upload a document that contains the answer."
                sources = []
            else:
                answer = generate_with_retry(client, build_prompt(question, [c for c, _ in results]))
                if answer is None:
                    answer = "The AI model is busy right now. Here are the most relevant passages instead."
                sources = results

        st.write(answer)
        if sources:
            show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})