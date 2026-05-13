# RAG local pentru documente in romana

Sistem RAG (Retrieval-Augmented Generation) complet local. Raspunde la intrebari in limba romana despre fisiere `.txt` preprocesate (OCR + LLM cleanup), cu citarea sursei.

**Stack:** Ollama + qwen2.5:14b · bge-m3 embeddings · ChromaDB · LlamaIndex · BM25 hybrid search · Streamlit

---

## Instalare pas cu pas (WSL2 / Ubuntu)

### 1. Instaleaza Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &          # porneste serverul in background
ollama pull qwen2.5:14b-instruct-q4_K_M
```

### 2. Verifica GPU-ul NVIDIA in WSL

```bash
nvidia-smi
# trebuie sa apara RTX 4070 Ti Super
```

Daca nu apare, instaleaza driverele NVIDIA cu suport WSL2 de pe [docs.nvidia.com/cuda/wsl-user-guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

### 3. Creaza venv si instaleaza dependentele

```bash
cd rag_romana/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# PyTorch cu CUDA (daca nu e instalat deja cu suport GPU):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. Adauga fisierele de date

```bash
cp /cale/catre/fisierele/tale/*.txt data/
```

### 5. Indexeaza documentele

```bash
python ingest.py
# Prima rulare descarca bge-m3 (~2GB) si poate dura cateva minute.
# Output asteptat: "Ingestie completa: X fisiere, Y chunks"
```

### 6. Porneste aplicatia

```bash
streamlit run app.py
# Deschide http://localhost:8501 in browser
```

---

## Sanity check rapid

1. Creeaza un fisier de test:
   ```bash
   echo "In 1971-1975, ritmul mediu anual de crestere a fost de 18.3%, mai rapid decat in perioada anterioara de 11.6%." > data/test.txt
   ```
2. `python ingest.py`
3. `streamlit run app.py`
4. Intreaba: *"In ce ani ritmul mediu anual de crestere a fost de 18.3%?"* — raspuns asteptat: **1971-1975**, sursa: `test.txt`
5. Intreaba: *"Cat a fost ritmul mediu anual in 1971-1975?"* — raspuns asteptat: **18.3%**

---

## Arhitectura retrieval

```
Query
  │
  ├─► bge-m3 embed ──► ChromaDB dense search (top-10)
  │
  └─► BM25 tokenize ──► BM25Okapi search (top-10)
                                │
                    Reciprocal Rank Fusion (k=60)
                                │
                           top-5 chunks
                                │
                    qwen2.5:14b (Ollama, streaming)
                                │
                          Raspuns + surse
```

Hybrid search este critic pentru intrebari cu cifre exacte: BM25 gaseste "18.3%" exact prin potrivire lexicala, in timp ce dense search gaseste context semantic.

---

## Scalare la 100+ fisiere (note pentru viitor, neimplementate)

- **Re-ranker cross-encoder**: adauga `llama-index-postprocessor-flag-embedding-reranker` sau `cross-encoder/ms-marco-MiniLM-L-6-v2` dupa RRF pentru a re-ranka top-5 cu un model mai precis.
- **Query rewriting**: genereaza 3-5 variante ale intrebarii cu LLM-ul si combina rezultatele (HyDE sau multi-query retrieval in LlamaIndex).
- **Migrare la Qdrant**: ChromaDB e bun pana la ~100k chunks; pentru volum mai mare, Qdrant ofera filtrare mai rapida si payload indexing.
- **Chunking adaptiv**: pentru documente cu tabele, foloseste un splitter care detecteaza structuri tabelate si le pastreaza intacte (ex. `MarkdownNodeParser`).
- **Cache semantic**: memoreaza raspunsurile pentru intrebari similare cu un cache vectorial (ex. `GPTCache`) pentru a reduce latenta.
