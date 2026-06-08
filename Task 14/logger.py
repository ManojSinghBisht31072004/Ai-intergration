import json
import logging
from datetime import datetime

# Setup Structured Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("RAG_Telemetry")

# 2026 Baseline Pricing Examples (e.g., GPT-4o or Claude 3.5 Sonnet equivalents)
INPUT_TOKEN_COST_PER_1K = 0.005
OUTPUT_TOKEN_COST_PER_1K = 0.015

def log_rag_transaction(query: str, response: str, latency: float, prompt_tokens: int, completion_tokens: int) -> dict:
    """
    Computes exact transaction pricing and logs performance metrics.
    """
    # Calculate costs
    input_cost = (prompt_tokens / 1000) * INPUT_TOKEN_COST_PER_1K
    output_cost = (completion_tokens / 1000) * OUTPUT_TOKEN_COST_PER_1K
    total_cost = input_cost + output_cost
    
    telemetry_payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "query_preview": query[:60],
        "metrics": {
            "latency_seconds": round(latency, 3),
            "target_met": latency < 3.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        },
        "financials": {
            "estimated_cost_usd": round(total_cost, 6)
        }
    }
    
    # Log structured JSON for log aggregators (Elastic/Splunk/CloudWatch)
    logger.info(json.dumps(telemetry_payload))
    return telemetry_payload