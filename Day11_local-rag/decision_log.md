Local RAG Pipeline

01 Model Selection
- Chosen LLM: qwen3:4b (Q4_K_M quant)
- Why:  
  Benchmarking showed that qwen3:4b delivered the best balance of:
  - response quality  
  - latency (~5–6 tokens/sec)  
  - stability with long prompts  
  - ability to follow structured instructions  

  qwen3:1.5b was faster but noticeably weaker on reasoning and multi‑sentence synthesis.  
  Larger models (7B+) were too slow on CPU and caused timeouts during generation.

- What I considered but rejected:  
  - qwen3:1.5b — fast but too shallow for multi‑document synthesis.  
  - llama3:8b / mistral‑7b — higher quality but too slow on local hardware, causing 120s+ timeouts.  
  - phi‑3‑mini — good for speed, but struggled with long‑context grounding and hallucinated more.

---

02 Embedding Model
- Chosen: nomic‑embed‑text
- Why:  
  - Very fast on CPU  
  - High recall for long enterprise‑style documents  
  - Stable vector dimensions  
  - Works well with multilingual content  
  I tested BGE-small and all‑MiniLM, but nomic produced more semantically relevant retrievals for policy‑style text.

---

03 Chunking Strategy
- Chunk size: 800 tokens  
- Overlap: 200 tokens  
- Why:  
  - Smaller chunks (300–400) caused fragmentation and loss of context, especially for tables and multi‑step procedures.  
  - Larger chunks (1000–1200) increased embedding time and sometimes exceeded model context limits.  
  - 800/200 gave the best balance: high recall, minimal context loss, and stable retrieval.

---

04 Retrieval Configuration
- Top‑K: 5
- Why:  
  - K=3 sometimes missed relevant sections when answers spanned multiple pages.  
  - K=8 introduced noise and diluted grounding.  
  - K=5 consistently returned the most relevant mix of chunks across all four documents.

---

05 Observations

i) What worked well
- Factual questions (e.g., “What is the purpose of the Information Security Policy?”) were answered perfectly because the relevant text was contained in a single chunk.
- Structured data (tables, metrics, SLAs) retrieved reliably after switching to pdfplumber.
- Cross‑document synthesis ( comparing ProjectHub vs Insight dashboards) worked well with Top‑K=5.

ii) What failed
- Couldnt use ChromaDB due to compatibility issue. 
- Unanswerable questions sometimes produced confident hallucinations when the retrieved chunks were weak.  
- Very long questions occasionally exceeded the LLM’s effective reasoning window.  
- Tables with multi‑column formatting sometimes extracted poorly, reducing retrieval accuracy.

iii) Local vs cloud expectations
- Azure OpenAI (GPT‑4o or GPT‑4 Turbo) would:  
  - produce more accurate synthesis  
  - hallucinate less on unanswerable questions  
  - handle long tables and multi‑page policies better  
  - respond 10–20× faster  
  But the local pipeline provides full privacy, zero API cost, and offline capability.

---

06 If I Had More Time / Better Hardware
- Switch to a 7B or 8B model (qwen2‑7b or mistral‑7b‑instruct) for higher accuracy.  
- Add reranking (e.g., bge‑reranker‑base) to improve retrieval precision.  
- Implement hybrid search (BM25 + embeddings) for policy documents.  
- Add chunk‑level metadata scoring to prioritize sections like “Purpose,” “Scope,” “SLA,” etc.  
- Move to persistent vector storage (Chroma or LanceDB) for faster reloads.  
- Add evaluation automation to compute precision, recall, and hallucination rate across all test questions.

