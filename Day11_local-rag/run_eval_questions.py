# run_eval_questions.py
from pipeline import answer_question
import numpy as np
import json

with open("test_questions.json") as f:
    test_questions = json.load(f)

results = []
for q in test_questions:
    output = answer_question(q["question"])
    results.append({
        "id": q["id"],
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "actual_answer": output["answer"],
        "retrieved_context": [c["content"] for c in output["retrieval"]["retrieved_chunks"]],
        "model": output["model"],
        "retrieval_time_ms": output["retrieval"]["retrieval_time_ms"],
        "generation_time_ms": output["generation_time_ms"],
    })

with open("pipeline_outputs_cloud.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Generated outputs for {len(results)} questions")