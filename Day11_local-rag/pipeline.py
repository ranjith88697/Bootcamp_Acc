from config_loader import load_config
from ingestion_retrieval import VectorStore, retrieve
from generation import generate_answer

# We create a global vector store so ingestion happens only once
vs = None

def initialize_pipeline():
    global vs
    if vs is None:
        cfg = load_config("config.yaml")
        vs = VectorStore(cfg)

        # Ingest only once
        from ingestion_retrieval import ingest
        ingest(cfg, vs)

def answer_question(question: str):
    initialize_pipeline()

    cfg = load_config("config.yaml")

    # 1. Retrieval
    retrieved = retrieve(question, cfg, vs)
    # retrieved is a list of (doc, meta) tuples

    retrieved_chunks = [
        {"content": d, "metadata": m}
        for d, m in retrieved
    ]

    # 2. Generation
    answer = generate_answer(question, retrieved, cfg)

    return {
        "answer": answer,
        "model": cfg["llm"]["model"],
        "retrieval": {
            "retrieved_chunks": retrieved_chunks,
            "retrieval_time_ms": 0  # your code does not measure time yet
        },
        "generation_time_ms": 0
    }
