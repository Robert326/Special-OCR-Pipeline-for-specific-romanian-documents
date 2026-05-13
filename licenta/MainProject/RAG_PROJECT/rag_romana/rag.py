import re
import pickle
import logging
from pathlib import Path
from typing import Generator

import torch
import requests
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

log = logging.getLogger(__name__)

STORAGE_DIR  = Path("storage")
BM25_PATH    = Path("bm25_index.pkl")
EMBED_MODEL  = "BAAI/bge-m3"
OLLAMA_URL   = "http://localhost:11434"
LLM_MODEL    = "qwen2.5:14b-instruct-q4_K_M"
LLM_FALLBACK = "llama3.1:8b-instruct-q4_K_M"
TOP_K        = 5
RRF_K        = 60  # constanta standard pentru Reciprocal Rank Fusion


def tokenize(text: str) -> list[str]:
    # aceeasi tokenizare ca in ingest.py — critica pentru consistenta BM25
    return re.findall(r'\w+(?:[.,]\w+)?', text.lower())


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _check_ollama() -> str:
    """Verifica daca Ollama ruleaza si alege modelul disponibil."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
        if LLM_MODEL in available:
            log.info("First Model Found: %s", LLM_MODEL)
            return LLM_MODEL
        if LLM_FALLBACK in available:
            log.warning("First Model %s unfound, using fallback %s", LLM_MODEL, LLM_FALLBACK)
            return LLM_FALLBACK
        log.error("No compatible model found in ollama. Run: ollama pull %s", LLM_MODEL)
        raise RuntimeError(f"No model available. Install with: ollama pull {LLM_MODEL}")
    except requests.ConnectionError:
        raise RuntimeError(
            f"Ollama does not respond {OLLAMA_URL}. "
            "Start: ollama serve"
        )


class RAGEngine:
    def __init__(self):
        device = _get_device()
        log.info("Device for embeddings: %s", device)

        self.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=device)

        # ChromaDB
        chroma_client = chromadb.PersistentClient(path=str(STORAGE_DIR))
        chroma_collection = chroma_client.get_or_create_collection("rag_romana")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_ctx, embed_model=self.embed_model
        )
        self.retriever = self.index.as_retriever(similarity_top_k=10)

        # BM25
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["index"]
        self.bm25_chunks = data["chunks"]

        # LLM
        model_name = _check_ollama()
        self.llm = Ollama(
            model=model_name,
            base_url=OLLAMA_URL,
            temperature=0.1,
            request_timeout=120,
        )
        self.doc_count = chroma_collection.count()
        log.info("RAGEngine gata. Documente indexate: %d", self.doc_count)

    def _rrf(self, dense_results: list, bm25_results: list) -> list[dict]:
        """Reciprocal Rank Fusion: combine dense scores and BM25."""
        scores: dict[str, float] = {}
        chunks_map: dict[str, dict] = {}

        # dense results: noduri LlamaIndex
        for rank, node in enumerate(dense_results):
            key = node.node.node_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunks_map[key] = {
                "text": node.node.text,
                "metadata": node.node.metadata,
            }

        # BM25 results: (chunk_dict, bm25_rank)
        for rank, (chunk, _) in enumerate(bm25_results):
            # genereaza un key unic din text (BM25 nu are node_id)
            key = f"bm25_{hash(chunk['text'])}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunks_map[key] = chunk

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [chunks_map[k] for k in sorted_keys[:TOP_K]]

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        # dense retrieval
        dense_nodes = self.retriever.retrieve(query)

        # BM25 retrieval
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        top10_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
        bm25_results = [(self.bm25_chunks[i], bm25_scores[i]) for i in top10_idx]

        return self._rrf(dense_nodes, bm25_results)

    def generate(self, query: str, chunks: list[dict]) -> Generator[str, None, None]:
        # construieste contextul cu tag-uri sursa
        context_parts = []
        for c in chunks:
            fname = c["metadata"].get("filename", "necunoscut")
            context_parts.append(f"[Sursa: {fname}]\n{c['text']}")
        context = "\n\n---\n\n".join(context_parts)

        prompt = (
            "Esti un asistent care raspunde la intrebari pe baza unor documente furnizate.\n"
            "Foloseste DOAR informatia din contextul de mai jos. "
            "Daca raspunsul nu se regaseste in context, spune sincer ca nu ai informatia.\n"
            "Raspunde in romana, clar si concis, dar include detaliile relevante (cifre, perioade, comparatii).\n"
            "La final mentioneaza sursele (fisierele) din care provine informatia.\n\n"
            f"Context:\n{context}\n\n"
            f"Intrebare: {query}\n\n"
            "Raspuns:"
        )

        # streaming prin Ollama
        response = self.llm.stream_complete(prompt)
        for token in response:
            yield token.delta

    def answer(self, query: str, top_k: int = TOP_K) -> dict:
        chunks = self.retrieve(query, top_k=top_k)
        full_answer = ""
        for token in self.generate(query, chunks):
            full_answer += token
        sources = list({c["metadata"].get("filename", "necunoscut") for c in chunks})
        return {
            "answer": full_answer,
            "sources": sources,
            "chunks_used": chunks,
        }
