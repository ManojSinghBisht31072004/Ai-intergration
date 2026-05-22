import json
from collections import defaultdict

def load_results(path="raw_results.json"):
    with open(path) as f:
        return json.load(f)

def aggregate(results):
    buckets = defaultdict(list)

    for r in results:
        buckets[r["model"]].append(r)

    summary = {}
    for model, records in buckets.items():
        n = len(records)
        summary[model] = {
            "total_requests":   n,
            "avg_latency_ms":   round(sum(r["latency_ms"]    for r in records) / n, 2),
            "avg_input_tokens": round(sum(r["input_tokens"]  for r in records) / n, 2),
            "avg_output_tokens":round(sum(r["output_tokens"] for r in records) / n, 2),
            "avg_cost_usd":     round(sum(r["cost_usd"]      for r in records) / n, 8),
            "total_cost_usd":   round(sum(r["cost_usd"]      for r in records),     6),
        }

    return summary

def print_summary(summary):
    print(f"\n{'Metric':<25}", end="")
    for model in summary:
        print(f"{model:<30}", end="")
    print()
    print("-" * (25 + 30 * len(summary)))

    metrics = [
        ("Total requests",    "total_requests",    ""),
        ("Avg latency (ms)",  "avg_latency_ms",    "ms"),
        ("Avg input tokens",  "avg_input_tokens",  ""),
        ("Avg output tokens", "avg_output_tokens", ""),
        ("Avg cost/req ($)",  "avg_cost_usd",      "$"),
        ("Total cost ($)",    "total_cost_usd",    "$"),
    ]

    for label, key, unit in metrics:
        print(f"{label:<25}", end="")
        for model in summary:
            val = summary[model][key]
            print(f"{val:<30}", end="")
        print()

if __name__ == "__main__":
    results = load_results()
    summary = aggregate(results)
    print_summary(summary)

    with open("aggregated_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to aggregated_summary.json")