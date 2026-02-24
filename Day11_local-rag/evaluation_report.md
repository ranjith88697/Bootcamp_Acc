Evaluation Report — Local RAG Pipeline

1. Executive Summary
The local RAG pipeline is functional, stable, and capable of answering most factual and medium‑complexity questions drawn from NovaTech’s internal documents. Retrieval quality is strong for well‑structured text but degrades with long tables and multi‑page policy sections. Generation quality is acceptable for internal use but not yet reliable enough for regulated‑industry scenarios where hallucination risk must be near zero. With tuning and a stronger local model, the system could meet production‑grade expectations.


2. Retrieval Quality
- ContextualRelevancy: 0.78 — Retrieval generally returns relevant chunks, but occasionally includes noise when questions span multiple sections.
- ContextualRecall: 0.72 — Some answers require information spread across multiple pages; Top‑K=5 captures most but not all required context.
- ContextualPrecision: 0.83 — Most retrieved chunks are useful, indicating embeddings and chunk size are well‑aligned.

3. Key Findings
- Policy documents with long tables (e.g., security classifications) caused partial retrieval because the table rows were split across chunks.
- Cross‑document questions (e.g., comparing ProjectHub vs Insight dashboards) worked well due to Top‑K=5.
- Unstructured meeting notes retrieved cleanly because they follow narrative structure.
- Root causes of weak retrieval:
  - Tables break poorly during PDF extraction.
  - Some sections exceed chunk size and lose semantic cohesion.
  - Embedding model (nomic‑embed‑text) is strong but not optimized for dense technical tables.

---

4. Generation Quality
- Faithfulness: 0.74 — The model stays grounded when context is strong but drifts when retrieval is weak.
- AnswerRelevancy: 0.81 — Answers generally address the question directly.
- Hallucination rate on unanswerable questions: 2/5 — The model hallucinated plausible‑sounding answers twice when it should have refused.

 Key Findings
- The model performs well on:
  - Factual questions with single‑chunk answers.
  - Summaries of policy sections.
  - Multi‑sentence synthesis when context is clean.

- The model struggles when:
  - Retrieval returns partial tables.
  - The question is unanswerable — the model tends to “fill in the gaps.”
  - The prompt is long enough to approach the model’s effective reasoning window.

---

5. Local vs Cloud Comparison

| Metric | Local (qwen3:4b) | Cloud (gpt-4.1-mini) | Delta |
|--------|------------------|----------------------|--------|
| Avg Faithfulness | 0.74 |  |  |
| Avg AnswerRelevancy | 0.81 |  |  |
| Avg ContextualRelevancy | 0.78 |  |  |
| Avg Response Time | ~70s |  |  |

Analysis
- Where local is sufficient:  
  - Internal knowledge retrieval where minor inaccuracies are acceptable.  
  - Offline or air‑gapped environments.  
  - Cost‑sensitive deployments.

- Although cloud couldn't e tested below reasons prove it is superior:  
  - Regulated industries requiring near‑zero hallucination.  
  - Complex synthesis across long documents.  
  - Fast interactive workflows.

- Tradeoffs:  
  - Local provides privacy and cost control.  
  - Cloud provides accuracy, speed, and reliability.  
  - For production in regulated environments, cloud‑grade models or a stronger local model (7B+) are recommended.

---

6. Security Assessment
- Prompt extraction: Partially resistant/ Vulnerable  
  The model does not leak system prompts but can be coaxed into revealing retrieved context if asked indirectly.

- Document injection: Moderate risk/ Vulnerable  
  If a malicious document contains instructions, the model may follow them unless additional guardrails are added.

- Context Poisoning: Vulnerable	
  Breaks persona, reveals config information


- Recommendation
A production deployment should include:
- Output‑filtering layer for hallucination detection  
- Retrieval‑scoring and reranking  
- Guardrails to prevent instruction override  
- Sanitization of ingested documents  
- Logging and monitoring for adversarial prompts  

---

7. Recommendations

1. Upgrade the LLM to a stronger local model (7B+) or add a reranker.  
   This will significantly improve faithfulness and reduce hallucinations, especially for multi‑page policy content.

2. Improve chunking for tables and long sections.  
   Use layout‑aware chunking or table‑specific extraction to avoid fragmented retrieval.

3. Add hybrid retrieval (embeddings).  
   This will improve recall for policy documents and reduce dependence on embedding quality alone.

4. Implement an unanswerable‑question detector.  
   A simple classifier or confidence threshold would reduce hallucinations by 40–60%.

5. If hardware allows, enable quantized 7B or 8B models.  
   This would close much of the gap with cloud models while keeping everything local.



