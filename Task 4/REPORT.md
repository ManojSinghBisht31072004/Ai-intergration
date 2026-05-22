# 📋 Project Report — AI Email Reply Generator API
## Day 4 Mini Project | Phase 1: LLM API Mastery

---

## 1. Project Overview

This project involved designing, building, and deploying a REST API that uses
Google Gemini AI to analyze emails. Given any email as input, the API returns:

- **Tone** — formal, neutral, urgent, or casual
- **Summary** — one sentence describing the email
- **Suggested Reply** — a ready-to-send professional response
- **Debug Info** — token count, latency, and cost per request

---

## 2. Architecture

```
Client (Thunder Client / curl / browser)
        │
        ▼
FastAPI Server (main.py)
        │
        ├── Input Validation (Pydantic models.py)
        │
        ├── Core Logic (analyzer.py)
        │     ├── Build prompt
        │     ├── Call Gemini API
        │     ├── Clean + Parse JSON response
        │     └── Return structured result
        │
        ├── Cost Tracking (cost_tracker.py)
        │     ├── Calculate USD cost
        │     └── Log to console
        │
        └── Response → JSON to client
```

---

## 3. API Contract

```
Endpoint : POST /email/analyze
Input    : { "email": "string (1–8000 chars)" }
Output   : {
             "tone": "formal|neutral|urgent|casual",
             "summary": "string",
             "suggestedReply": "string",
             "debug": {
               "input_tokens": int,
               "output_tokens": int,
               "total_tokens": int,
               "latency_ms": float,
               "cost_usd": float
             }
           }
```

---

## 4. Model Selection

| Model Tried | Result |
|-------------|--------|
| `gemini-2.0-flash-lite` | ❌ Free tier daily quota exceeded (200/day) |
| `gemini-1.5-flash` | ❌ Not found on this API key version |
| `gemini-3.1-flash-lite` | ✅ Working — fast, stable, low cost |

**Final Model:** `gemini-3.1-flash-lite`

---

## 5. Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| `ModuleNotFoundError: models` | File was named `model.py` → renamed to `models.py` |
| `429 Rate Limit` on flash-lite | Switched to `gemini-3.1-flash-lite` (higher quota) |
| `404 model not found` | Used `list_models()` to find available models |
| JSON parse error on Hindi email | Built character-by-character JSON cleaner in `analyzer.py` |
| Latency > 3s on first request | Expected cold start — subsequent requests under 2s |

---

## 6. Cost Analysis

**Model:** Gemini 3.1 Flash Lite

| Token Type | Price |
|------------|-------|
| Input | $0.075 / 1M tokens |
| Output | $0.300 / 1M tokens |

**Average per request (from 20 test cases):**

| Metric | Value |
|--------|-------|
| Avg Input Tokens | ~250 |
| Avg Output Tokens | ~85 |
| Avg Total Tokens | ~335 |
| Avg Cost | ~$0.000044 |
| Avg Latency | ~1200ms |

**Formula:**
```
cost = (input_tokens × $0.000000075) + (output_tokens × $0.000000300)
```

---

## 7. Test Cases Summary

| Category | Count | Pass Rate |
|----------|-------|-----------|
| Formal emails | 5 | 100% |
| Casual emails | 3 | 100% |
| Urgent emails | 4 | 100% |
| Neutral emails | 4 | 100% |
| Non-English (Hindi, Spanish) | 2 | 100% |
| Edge cases | 2 | 100% |
| **Total** | **20** | **100%** |

---

## 8. Acceptance Criteria Status

| Criteria | Status |
|----------|--------|
| Endpoint returns tone, summary, suggestedReply | ✅ |
| Response time < 3s for typical emails | ✅ |
| Cost per request calculated and logged | ✅ |
| 20 test cases documented | ✅ |
| Invalid/oversized input returns clear error | ✅ |
| Deployed with live URL | ✅ |
| API key in env, not in code | ✅ |

---

## 9. Key Learnings

1. **Structured output from LLMs** requires defensive parsing —
   models sometimes return markdown fences or raw newlines inside JSON.

2. **Free tier limits** vary drastically between models —
   always check `list_models()` and daily quotas before choosing.

3. **Pydantic + FastAPI** make input validation and response shaping
   extremely clean with minimal boilerplate.

4. **Cost tracking per request** is essential for production APIs —
   even tiny costs add up at scale.

5. **Multilingual support** works out of the box with Gemini —
   no extra configuration needed.

---

## 10. Future Improvements

| Improvement | Benefit |
|-------------|---------|
| Add rate limiting (SlowAPI) | Prevent abuse |
| Add request logging to file | Audit trail |
| Add authentication (API key header) | Security |
| Cache repeated emails (Redis) | Reduce cost |
| Add confidence score to tone | Better insights |
| Support email threads (array input) | More realistic use case |

---

## 11. Live Deployment

| Item | Detail |
|------|--------|
| Platform | Render (Free Tier) |
| Live URL | https://email-reply-api.onrender.com |
| Docs URL | https://email-reply-api.onrender.com/docs |
| Health URL | https://email-reply-api.onrender.com/health |

---

*Report generated: May 2026 | Day 4 – Mini Project 1*