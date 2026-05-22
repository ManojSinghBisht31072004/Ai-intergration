import os
import json
import time
from groq import Groq
from dotenv import load_dotenv
from prompts import PROMPTS

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODELS = {
    "llama-3.3-70b-versatile": {
        "input_price_per_1m":  0.59,
        "output_price_per_1m": 0.79,
    },
    "llama-3.1-8b-instant": {
        "input_price_per_1m":  0.05,
        "output_price_per_1m": 0.08,
    },
}

def compute_cost(model_name, input_tokens, output_tokens):
    pricing = MODELS[model_name]
    input_cost  = (input_tokens  / 1_000_000) * pricing["input_price_per_1m"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_price_per_1m"]
    return round(input_cost + output_cost, 8)

def run_benchmark():
    results = []

    for prompt in PROMPTS:
        for model_name in MODELS:
            print(f"Running prompt {prompt['id']} | model: {model_name} ...")

            start = time.perf_counter()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt["text"]}],
                temperature=0.7,
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            input_tokens  = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            output_text   = response.choices[0].message.content

            cost = compute_cost(model_name, input_tokens, output_tokens)

            results.append({
                "prompt_id":      prompt["id"],
                "category":       prompt["category"],
                "prompt":         prompt["text"],
                "model":          model_name,
                "latency_ms":     latency_ms,
                "input_tokens":   input_tokens,
                "output_tokens":  output_tokens,
                "total_tokens":   input_tokens + output_tokens,
                "cost_usd":       cost,
                "response":       output_text,
            })

    with open("raw_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone. Results saved to raw_results.json")

if __name__ == "__main__":
    run_benchmark()