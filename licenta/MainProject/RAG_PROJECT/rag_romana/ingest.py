import os
import re
import time
import pickle
import logging
from pathlib import Path

import torch
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rank_bm25 import BM25Okapi
import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR     = Path("data")
STORAGE_DIR  = Path("storage")
BM25_PATH    = Path("bm25_index.pkl")
EMBED_MODEL  = "BAAI/bge-m3"
CHUNK_SIZE   = 300
CHUNK_OVERLAP = 50


def tokenize(text: str) -> list[str]:
    # keeps decimal numbers like 18.3 and 1.234 as single tokens
    return re.findall(r'\w+(?:[.,]\w+)?', text.lower())


def get_device() -> str:
    if torch.cuda.is_available():
        log.info("CUDA available: %s", torch.cuda.get_device_name(0))
        return "cuda"
    log.warning("CUDA unavailable — embeddings will run on CPU (be patient :D)")
    return "cpu"


def ask_reindex() -> bool:
    ans = input("There is already an Index found. Do you want to reindex from zero? [y/N]: ").strip().lower()
    return ans == "y"


def extract_page(filename: str) -> int | None:
    # extrage numarul de pagina din pattern: llm_Poza_page_3.txt -> 3
    m = re.search(r'page[_-]?(\d+)', filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


def main():
    # verifica ca exista fisiere de procesat
    txt_files = list(DATA_DIR.glob("*.txt"))
    if not txt_files:
        log.error("There are no files .txt in %s", DATA_DIR)
        return

    # daca indexul exista, intreaba userul
    if STORAGE_DIR.exists() and BM25_PATH.exists():
        if not ask_reindex():
            log.info("Aborted reindexation")
            return
        # stergere index vechi
        import shutil
        shutil.rmtree(STORAGE_DIR)
        BM25_PATH.unlink()

    STORAGE_DIR.mkdir(exist_ok=True)
    log.info("Found %d files .txt in %s", len(txt_files), DATA_DIR)

    # incarca documentele
    log.info("Loading the documents")
    reader = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".txt"],
        encoding="utf-8",
    )
    documents = reader.load_data()

    # splitter: 300 tokeni, overlap 50
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)

    # adauga metadata: filename, chunk_id, page
    for i, node in enumerate(nodes):
        fname = Path(node.metadata.get("file_name", "")).name
        node.metadata["filename"] = fname
        node.metadata["chunk_id"] = i
        page = extract_page(fname)
        if page is not None:
            node.metadata["page"] = page

    log.info("Total chunks: %d", len(nodes))

    # embeddings pe GPU daca e disponibil
    device = get_device()
    log.info("Downloading embeddings model %s (~2GB at first run)...", EMBED_MODEL)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=device)

    # ChromaDB persistent
    chroma_client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    chroma_collection = chroma_client.get_or_create_collection("rag_romana")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection) # adaptor intre LlamaIndex si ChromaDB
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store) # foloseste ChromaDB ca storage pentru vectori, restul (docstore, index store) lasa le asa

    log.info("Calculating embeddings and saving them in ChromaDB...")
    t0 = time.time()
    VectorStoreIndex(nodes, storage_context=storage_ctx, embed_model=embed_model, show_progress=True)
    log.info("ChromaDB populated in %.1fs", time.time() - t0)

    # construieste indexul BM25 peste aceleasi chunks (asta e pentru cuvinte cheie)
    log.info("Building the index BM25...")
    bm25_corpus = []
    bm25_chunks = []
    for node in nodes:
        tokens = tokenize(node.text)
        bm25_corpus.append(tokens)
        bm25_chunks.append({
            "text": node.text,
            "metadata": node.metadata,
        })

    # lucreaza cu statistici lungimi scoruri de relevanta
    bm25_index = BM25Okapi(bm25_corpus)
    
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"index": bm25_index, "chunks": bm25_chunks}, f)

    log.info("BM25 salvat in %s", BM25_PATH)
    log.info("Ingestie completa: %d fisiere, %d chunks, %.1fs total",
             len(txt_files), len(nodes), time.time() - t0)


if __name__ == "__main__":
    main()
