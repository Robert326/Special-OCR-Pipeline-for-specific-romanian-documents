import logging
import subprocess
import sys
from pathlib import Path

import streamlit as st

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Asistent documente — RAG local", layout="wide")

BM25_PATH   = Path("bm25_index.pkl")
STORAGE_DIR = Path("storage")


@st.cache_resource(show_spinner="Se incarca modelele (prima rulare poate dura cateva minute)...")
def load_engine():
    from rag import RAGEngine
    return RAGEngine()


def index_exists() -> bool:
    return BM25_PATH.exists() and STORAGE_DIR.exists()


def run_ingest():
    with st.spinner("Indexare in curs..."):
        result = subprocess.run(
            [sys.executable, "ingest.py"],
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        st.success("Indexare completa!")
        st.cache_resource.clear()
        st.rerun()
    else:
        st.error(f"Eroare la indexare:\n{result.stderr}")


# Sidebar 

with st.sidebar:
    st.title("Configurare")

    if index_exists():
        st.success("Index gasit")
        try:
            engine = load_engine()
            st.info(f"Documente indexate: {engine.doc_count}")
        except Exception as e:
            st.error(str(e))
            engine = None
    else:
        st.warning("Index negasit. Adauga fisiere in data/ si indexeaza.")
        engine = None

    top_k = st.slider("Chunks returnate (top_k)", min_value=1, max_value=10, value=5)

    if st.button("Reindexeaza"):
        run_ingest()

    if st.button("Curata conversatia"):
        st.session_state.messages = []
        st.rerun()

#  Chat principal 

st.title("Asistent documente — RAG local")

if "messages" not in st.session_state:
    st.session_state.messages = []

# afiseaza istoricul
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "chunks" in msg:
            with st.expander("Surse"):
                for chunk in msg["chunks"]:
                    fname = chunk["metadata"].get("filename", "necunoscut")
                    page  = chunk["metadata"].get("page", "")
                    label = f"**{fname}**" + (f" — pagina {page}" if page else "")
                    st.markdown(label)
                    st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))

# input utilizator
if query := st.chat_input("Pune o intrebare despre documente..."):
    if engine is None:
        st.error("Indexul nu e incarcat. Verifica sidebar-ul.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            chunks = engine.retrieve(query, top_k=top_k)

            # streaming raspuns
            response_placeholder = st.empty()
            full_response = ""
            for token in engine.generate(query, chunks):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

            # surse
            with st.expander("Surse"):
                for chunk in chunks:
                    fname = chunk["metadata"].get("filename", "necunoscut")
                    page  = chunk["metadata"].get("page", "")
                    label = f"**{fname}**" + (f" — pagina {page}" if page else "")
                    st.markdown(label)
                    st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "chunks": chunks,
        })
