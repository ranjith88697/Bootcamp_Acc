import requests

def build_prompt(question: str, context_chunks):
    context_text = "\n\n".join([f"[Source: {m['source']} p.{m['page']}] {d}" for d, m in context_chunks])

    return f"""
You are a helpful assistant. Use ONLY the information in the context below to answer the question.
If the answer is not in the context, say "The document does not contain the answer."

### Context:
{context_text}

### Question:
{question}

### Answer:
"""


def generate_answer(question: str, context_chunks, cfg: dict):
    prompt = build_prompt(question, context_chunks)

    llm_cfg = cfg["llm"]
    url = f"{llm_cfg['base_url']}/chat/completions"

    payload = {
        "model": llm_cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": llm_cfg["temperature"],
        "max_tokens": llm_cfg["max_tokens"]
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()

    data = resp.json()
    return data["choices"][0]["message"]["content"]
