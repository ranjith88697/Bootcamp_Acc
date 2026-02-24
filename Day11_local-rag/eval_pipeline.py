import os
import json
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models import AzureOpenAIModel
from deepeval.evaluate import AsyncConfig  # Fixed import path

# -----------------------------
# 1. Setup Azure OpenAI Judge
# -----------------------------
os.environ["AZURE_OPENAI_API_KEY"] = "################"

JUDGE = AzureOpenAIModel(
    model="gpt-4.1-mini",
    deployment_name="gpt-4.1-mini",
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version="2025-01-01-preview",
    base_url="https://ds-ai-internship.openai.azure.com",
)

# -----------------------------
# 2. Load and Clean Pipeline Outputs
# -----------------------------
with open("pipeline_outputs_local.json") as f:
    results = json.load(f)

test_cases = []

for i, r in enumerate(results):
    actual = r.get("actual_answer")
    
    # Fix: Ensure actual_output is not empty to avoid MissingTestCaseParamsError
    if actual and str(actual).strip():
        tc = LLMTestCase(
            input=r.get("question", ""),
            actual_output=actual,
            expected_output=r.get("expected_answer"),
            retrieval_context=r.get("retrieved_context", []),
        )
        test_cases.append(tc)
    else:
        print(f"⚠️ Row {i} skipped: 'actual_answer' is missing or empty.")

# -----------------------------
# 3. Define Metrics with Azure Judge
# -----------------------------
metrics = [
    FaithfulnessMetric(threshold=0.7, include_reason=True, model=JUDGE),
    AnswerRelevancyMetric(threshold=0.7, include_reason=True, model=JUDGE),
    ContextualRelevancyMetric(threshold=0.7, include_reason=True, model=JUDGE),
    ContextualRecallMetric(threshold=0.7, include_reason=True, model=JUDGE),
    ContextualPrecisionMetric(threshold=0.7, include_reason=True, model=JUDGE),
]

# Configure throttling to prevent 429 Rate Limit errors on Azure
async_config = AsyncConfig(max_concurrent=5, throttle_value=1)

# -----------------------------
# 4. Run Evaluation
# -----------------------------
if test_cases:
    # Run evaluation
    eval_results = evaluate(
        test_cases=test_cases, 
        metrics=metrics, 
        async_config=async_config
    )

    # -----------------------------
    # 5. Extract and Save Results (Fix for 'to_dict' Error)
    # -----------------------------
    final_output = []
    
    # Iterate through the TestResult objects inside eval_results
    for result in eval_results.test_results:
        metrics_summary = []
        for metric in result.metrics_data:
            metrics_summary.append({
                "metric_name": metric.name,
                "score": metric.score,
                "reason": metric.reason,
                "success": metric.success
            })
            
        final_output.append({
            "input": result.input,
            "actual_output": result.actual_output,
            "success": result.success,
            "metrics": metrics_summary
        })

    # Write the cleaned data to JSON
    with open("eval_results_cloud.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n✅ Evaluation complete. Processed {len(test_cases)} cases.")
    print("Results saved to: eval_results_local.json")
else:
    print("\n❌ No valid test cases found to evaluate.")