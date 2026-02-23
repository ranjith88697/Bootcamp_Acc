import os
import numpy as np
import requests
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# -----------------------------
# Embedding Client
# -----------------------------
class EmbeddingClient:
    def __init__(self, cfg: dict):
        emb_cfg = cfg["embeddings"]
        self.model = emb_cfg["model"]
        self.base_url = emb_cfg["base_url"]
        print(">>> Embedding model in use:", self.model)

    def embed(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        all_embeddings = []
        url = f"{self.base_url}/embeddings"

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            payload = {"model": self.model, "input": batch}

            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()

            data = resp.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


# -----------------------------
# Simple In-Memory Vector Store
# -----------------------------
class VectorStore:
    def __init__(self, cfg: dict):
        self.embeddings: List[List[float]] = []
        self.documents: List[str] = []
        self.metadatas: List[dict] = []

    def add(self, embeddings, docs, metas):
        self.embeddings.extend(embeddings)
        self.documents.extend(docs)
        self.metadatas.extend(metas)

    def query(self, query_emb, top_k):
        if not self.embeddings:
            return {"documents": [[]], "metadatas": [[]]}

        embs = np.array(self.embeddings, dtype="float32")
        q = np.array(query_emb[0], dtype="float32")

        scores = embs @ q / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q) + 1e-8)
        idx = np.argsort(scores)[::-1][:top_k]

        docs = [self.documents[i] for i in idx]
        metas = [self.metadatas[i] for i in idx]

        return {"documents": [docs], "metadatas": [metas]}


# -----------------------------
# PDF Loading + Chunking
# -----------------------------
import pdfplumber

def load_pdfs(pdf_dir: str):
    docs = []
    metas = []

    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, fname)

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                docs.append(text)
                metas.append({"source": fname, "page": i + 1})

    return docs, metas


def chunk_documents(docs, metas, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    chunk_metas = []

    for doc, meta in zip(docs, metas):
        parts = splitter.split_text(doc)
        for p in parts:
            chunks.append(p)
            chunk_metas.append(meta)

    return chunks, chunk_metas


# -----------------------------
# Ingestion
# -----------------------------
def ingest(cfg: dict, vs: VectorStore):
    pdf_dir = cfg["ingestion"]["pdf_dir"]
    chunk_size = cfg["ingestion"]["chunk_size"]
    chunk_overlap = cfg["ingestion"]["chunk_overlap"]

    print("Loading PDFs...")
    docs, metas = load_pdfs(pdf_dir)

    print("Chunking...")
    chunks, chunk_metas = chunk_documents(docs, metas, chunk_size, chunk_overlap)

    print("Embedding chunks...")
    emb_client = EmbeddingClient(cfg)
    embeddings = emb_client.embed(chunks)

    print("Storing in memory...")
    vs.add(embeddings, chunks, chunk_metas)

    print("Ingestion complete.")
    print("Number of chunks:", len(chunks))



# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query: str, cfg: dict, vs: VectorStore):
    top_k = cfg["retrieval"]["top_k"]

    emb_client = EmbeddingClient(cfg)
    query_emb = emb_client.embed([query])

    results = vs.query(query_emb, top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return list(zip(docs, metas))
    print("Retrieved docs:", docs)

