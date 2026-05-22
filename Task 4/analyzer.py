import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

from models import EmailResponse, DebugInfo
from cost_tracker import calculate_cost, log_usage

load_dotenv()

# ─── Configure Gemini ──────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ─── System Prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert email analyst. Analyze the given email and respond ONLY with a valid JSON object.

The JSON must have exactly these three fields:
1. "tone"           : one of "formal", "neutral", "urgent", or "casual"
2. "summary"        : a single sentence summarizing what the email is about
3. "suggestedReply" : a professional, complete, ready-to-send reply to the email

Rules:
- Return ONLY the raw JSON. No markdown, no code blocks, no explanation.
- suggestedReply must be polite, clear, and context-aware.
- If the email is in another language, detect and match the same language in suggestedReply.

Example output format:
{
  "tone": "formal",
  "summary": "The sender is requesting a meeting next Monday.",
  "suggestedReply": "Dear [Name],\\n\\nThank you for reaching out..."
}
"""


def analyze_email(email_text: str) -> EmailResponse:
    """
    Send email text to Gemini and return structured analysis.

    Args:
        email_text : Raw email string (already validated for length)

    Returns:
        EmailResponse with tone, summary, suggestedReply, and debug info

    Raises:
        ValueError : If Gemini returns unparseable JSON
        RuntimeError: If Gemini API call fails
    """

    model = genai.GenerativeModel(
        model_name="gemini-3.1-flash-lite",
        generation_config=genai.GenerationConfig(
            temperature=0.4,           # Slight creativity, mostly consistent
            max_output_tokens=1024,    # Enough for a detailed reply
        )
    )

    prompt = f"{SYSTEM_PROMPT}\n\nEmail to analyze:\n\"\"\"\n{email_text}\n\"\"\""

    # ─── API Call with Timing ──────────────────────────────
    start_time = time.time()

    try:
        response = model.generate_content(prompt)
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {str(e)}")

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # ─── Extract Token Usage ───────────────────────────────
    usage = response.usage_metadata
    input_tokens  = usage.prompt_token_count
    output_tokens = usage.candidates_token_count
    total_tokens  = usage.total_token_count

    # ─── Calculate Cost ────────────────────────────────────
    cost = calculate_cost(input_tokens, output_tokens)
    log_usage(input_tokens, output_tokens, latency_ms, cost)

    # ─── Parse JSON Response ───────────────────────────────
    raw_text = response.text.strip()

    # Remove markdown code fences if Gemini adds them
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Gemini response as JSON: {str(e)}\nRaw: {raw_text}")

    # ─── Validate Required Fields ──────────────────────────
    required_fields = {"tone", "summary", "suggestedReply"}
    missing = required_fields - parsed.keys()
    if missing:
        raise ValueError(f"Gemini response missing fields: {missing}")

    # ─── Normalize Tone ────────────────────────────────────
    valid_tones = {"formal", "neutral", "urgent", "casual"}
    tone = parsed["tone"].lower().strip()
    if tone not in valid_tones:
        tone = "neutral"   # Fallback safely

    # ─── Build and Return Response ─────────────────────────
    debug = DebugInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        cost_usd=cost
    )

    return EmailResponse(
        tone=tone,
        summary=parsed["summary"],
        suggestedReply=parsed["suggestedReply"],
        debug=debug
    )