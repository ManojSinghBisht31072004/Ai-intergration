import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from prompts import build_crm_prompt

load_dotenv()

# Load all 3 keys — skip empty ones
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
]
API_KEYS = [key for key in API_KEYS if key.strip()]

if not API_KEYS:
    raise ValueError("No Gemini API keys found in .env file")


def get_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.3,
        ),
    )


def analyze_lead(name: str, company: str, notes: str) -> dict:
    prompt = build_crm_prompt(name, company, notes)

    last_error = None

    # Try each API key one by one
    for index, api_key in enumerate(API_KEYS):
        try:
            print(f"[INFO] Trying API key {index + 1} of {len(API_KEYS)}...")
            model = get_model(api_key)
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

            # Strip markdown fences if present
            if "```" in raw_text:
                lines = raw_text.splitlines()
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw_text = "\n".join(lines).strip()

            parsed = json.loads(raw_text)

            # Validate required keys
            required_keys = {"summary", "suggestedFollowUp", "sentimentScore"}
            if not required_keys.issubset(parsed.keys()):
                raise ValueError(f"Missing keys: {list(parsed.keys())}")

            # Sanitize sentimentScore
            if parsed["sentimentScore"] not in ("positive", "neutral", "negative"):
                parsed["sentimentScore"] = "neutral"

            print(f"[SUCCESS] API key {index + 1} worked!")
            return parsed

        except Exception as e:
            error_msg = str(e)
            print(f"[WARN] API key {index + 1} failed: {error_msg[:80]}...")
            last_error = error_msg

            # Only retry on quota/auth errors
            # Stop retrying on JSON or logic errors
            if "429" in error_msg or "400" in error_msg or "403" in error_msg or "API_KEY_INVALID" in error_msg:
                continue
            else:
                raise RuntimeError(f"Non-retryable error: {error_msg}")

    # All keys failed
    raise RuntimeError(
        f"All {len(API_KEYS)} API keys exhausted. Last error: {last_error}"
    )