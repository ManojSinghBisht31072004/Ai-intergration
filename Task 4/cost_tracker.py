# Gemini 2.0 Flash Lite Pricing
# Input:  $0.075 per 1,000,000 tokens
# Output: $0.300 per 1,000,000 tokens

PRICE_INPUT_PER_TOKEN  = 0.075 / 1_000_000   # $ per input token
PRICE_OUTPUT_PER_TOKEN = 0.300 / 1_000_000   # $ per output token


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost in USD for a single Gemini API request.

    Args:
        input_tokens  : Number of tokens in the prompt
        output_tokens : Number of tokens in the response

    Returns:
        cost in USD (float, rounded to 8 decimal places)
    """
    cost = (input_tokens * PRICE_INPUT_PER_TOKEN) + \
           (output_tokens * PRICE_OUTPUT_PER_TOKEN)
    return round(cost, 8)


def log_usage(input_tokens: int, output_tokens: int,
              latency_ms: float, cost: float) -> None:
    """
    Print token usage and cost to console/logs.
    """
    print(f"[USAGE] Input tokens   : {input_tokens}")
    print(f"[USAGE] Output tokens  : {output_tokens}")
    print(f"[USAGE] Total tokens   : {input_tokens + output_tokens}")
    print(f"[USAGE] Latency        : {latency_ms:.1f} ms")
    print(f"[USAGE] Cost           : ${cost:.8f} USD")
    print("-" * 45)