import json
import time
from openai import OpenAI
import os

# -----------------------------
# Configuration
# -----------------------------
MODELS = [
    "qwen3:4b",
    "qwen3:1.5b"
]

BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"

PROMPTS = [
    {"category": "factual", "prompt": "What causes tides on Earth? Answer in 2–3 sentences."},
    {"category": "reasoning", "prompt": "A train travels 120 km in 1.5 hours and then 80 km in 1 hour. What is its average speed for the entire trip?"},
    {"category": "summarization", "prompt": (
        "Summarize the following paragraph in 2 sentences:\n\n"
        "Artificial intelligence is transforming industries by enabling automation, "
        "enhancing decision‑making, and unlocking new capabilities. Companies across "
        "healthcare, finance, and manufacturing are adopting AI to improve efficiency "
        "and deliver better outcomes."
    )},
    {"category": "structured", "prompt": (
        'List 3 European capitals. Respond ONLY with valid JSON in this format: '
        '[{"city": "...", "country": "..."}]'
    )},
    {"category": "code", "prompt": "Write a Python function that checks if a number is prime."},
    {"category": "instruction", "prompt": (
        "Provide a 3‑step plan for improving team communication. "
        "Each step must be exactly one sentence."
    )}
]

OUTPUT_FILE = "benchmark_results.json"


# -----------------------------
# Benchmark runner
# -----------------------------
def run_benchmark(model: str, prompt: str) -> dict:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    elapsed = time.time() - start

    content = response.choices[0].message.content
    usage = getattr(response, "usage", None)

    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

    tokens_per_second = (
        round(completion_tokens / elapsed, 2)
        if completion_tokens
        else None
    )

    result = {
        "model": model,
        "prompt": prompt,
        "response": content,
        "time_seconds": round(elapsed, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_second,
    }

    print(f"\n[{model}] {prompt[:50]}...  →  {round(elapsed, 2)}s")

    return result


# -----------------------------
# Main
# -----------------------------
def main():
    all_results = []

    # If file exists, load previous results
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            all_results = json.load(f)

    for model in MODELS:
        for item in PROMPTS:
            result = run_benchmark(model, item["prompt"])
            result["category"] = item["category"]
            all_results.append(result)

            # Write after each benchmark
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_results, f, indent=2)

            print("✓ Saved partial results")

    print("\nBenchmark complete! Full results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
