"""
Benchmark script to evaluate Image Search performance
Metrics: Speed, Accuracy, Score Quality
"""
import time
import numpy as np
from pathlib import Path
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

# Test queries with expected relevant keywords
TEST_QUERIES = [
    {"query": "cat sleeping", "expected_keywords": ["cat"]},
    {"query": "cute cat", "expected_keywords": ["cat"]},
    {"query": "cat playing", "expected_keywords": ["cat"]},
    {"query": "fluffy cat", "expected_keywords": ["cat"]},
    {"query": "kitten", "expected_keywords": ["cat"]},
]

def benchmark_search(query: str, top_k: int = 5):
    """Measure search performance"""
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/api/search",
        json={"query": query, "top_k": top_k}
    )
    end = time.time()
    
    return {
        "query": query,
        "response_time_ms": (end - start) * 1000,
        "results": response.json()["results"] if response.ok else [],
        "status": response.status_code
    }

def evaluate_relevance(results, expected_keywords):
    """Check if results contain expected keywords"""
    relevant_count = 0
    for r in results:
        img_id = r["image_id"].lower()
        if any(kw in img_id for kw in expected_keywords):
            relevant_count += 1
    return relevant_count / len(results) if results else 0

def calculate_mrr(results, expected_keywords):
    """Mean Reciprocal Rank - position of first relevant result"""
    for idx, r in enumerate(results):
        img_id = r["image_id"].lower()
        if any(kw in img_id for kw in expected_keywords):
            return 1.0 / (idx + 1)
    return 0.0

print("="*60)
print("IMAGE SEARCH BENCHMARK")
print("="*60)

all_response_times = []
all_precisions = []
all_mrrs = []
all_scores = []

for test in TEST_QUERIES:
    result = benchmark_search(test["query"], top_k=5)
    
    if result["status"] != 200:
        print(f"\nERROR: {test['query']} - Status {result['status']}")
        continue
    
    precision = evaluate_relevance(result["results"], test["expected_keywords"])
    mrr = calculate_mrr(result["results"], test["expected_keywords"])
    scores = [r["score"] for r in result["results"]]
    
    all_response_times.append(result["response_time_ms"])
    all_precisions.append(precision)
    all_mrrs.append(mrr)
    all_scores.extend(scores)
    
    print(f"\nQuery: '{test['query']}'")
    print(f"  Response Time: {result['response_time_ms']:.1f}ms")
    print(f"  Precision@5: {precision:.2%}")
    print(f"  MRR: {mrr:.3f}")
    print(f"  Top score: {scores[0]:.4f}")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")

print("\n" + "="*60)
print("OVERALL METRICS")
print("="*60)
print(f"Average Response Time: {np.mean(all_response_times):.1f}ms")
print(f"Min/Max Response Time: {min(all_response_times):.1f}ms / {max(all_response_times):.1f}ms")
print(f"\nAverage Precision@5: {np.mean(all_precisions):.2%}")
print(f"Average MRR: {np.mean(all_mrrs):.3f}")
print(f"\nScore Statistics:")
print(f"  Mean: {np.mean(all_scores):.4f}")
print(f"  Std: {np.std(all_scores):.4f}")
print(f"  Range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

avg_time = np.mean(all_response_times)
avg_precision = np.mean(all_precisions)
avg_mrr = np.mean(all_mrrs)

if avg_time < 200:
    print("Speed: EXCELLENT (<200ms)")
elif avg_time < 500:
    print("Speed: GOOD (<500ms)")
else:
    print("Speed: NEEDS IMPROVEMENT (>500ms)")

if avg_precision > 0.8:
    print("Accuracy: EXCELLENT (>80% precision)")
elif avg_precision > 0.6:
    print("Accuracy: GOOD (>60% precision)")
else:
    print("Accuracy: NEEDS IMPROVEMENT (<60% precision)")

if avg_mrr > 0.8:
    print("Relevance: EXCELLENT (first result usually correct)")
elif avg_mrr > 0.5:
    print("Relevance: GOOD (relevant in top 2-3)")
else:
    print("Relevance: NEEDS IMPROVEMENT")
