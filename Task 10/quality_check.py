import json

# Keyword hints per category — response should contain at least one
QUALITY_HINTS = {
    "reasoning":     ["because", "therefore", "due to", "as a result", "explains", "reason"],
    "code":          ["def ", "return", "function", "python", "```"],
    "summarization": ["•", "-", "1.", "first", "second", "key", "difference", "main"],
    "creative":      [" ", "\n"],  # just needs non-empty content
}

MIN_LENGTH = {
    "reasoning":     80,
    "code":          50,
    "summarization": 50,
    "creative":      30,
}

def score_response(record):
    category = record["category"]
    response = record["response"].lower()

    hints    = QUALITY_HINTS.get(category, [])
    min_len  = MIN_LENGTH.get(category, 30)

    length_ok  = len(record["response"]) >= min_len
    keyword_ok = any(kw in response for kw in hints)

    score = 1 if (length_ok and keyword_ok) else 0

    return {
        "prompt_id":   record["prompt_id"],
        "category":    category,
        "model":       record["model"],
        "score":       score,
        "length_ok":   length_ok,
        "keyword_ok":  keyword_ok,
        "response_len": len(record["response"]),
    }

def run_quality_check(path="raw_results.json"):
    with open(path) as f:
        results = json.load(f)

    scored = [score_response(r) for r in results]

    # Summary per model
    models = list({r["model"] for r in scored})
    print(f"\n{'Model':<35} {'Score':>10} {'Out of':>8}")
    print("-" * 55)
    for model in models:
        model_scores = [s for s in scored if s["model"] == model]
        total = sum(s["score"] for s in model_scores)
        print(f"{model:<35} {total:>10} {len(model_scores):>8}")

    with open("quality_scores.json", "w") as f:
        json.dump(scored, f, indent=2)

    print("\nDetailed scores saved to quality_scores.json")
    return scored

if __name__ == "__main__":
    run_quality_check()