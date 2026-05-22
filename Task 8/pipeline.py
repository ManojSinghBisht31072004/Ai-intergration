# # pipeline.py
# # import json
# # import time
# # import google.generativeai as genai
# # from config import get_next_key, MODEL_NAME
# # from db import insert_ticket, insert_failed_ticket

# from groq import Groq
# from config import GROQ_API_KEY, DB_FILE
# from db import insert_ticket, insert_failed_ticket
# import json, time
# from groq import Groq  # USing grok api key

# client = Groq(api_key=GROQ_API_KEY)

# def call_gemini(prompt: str) -> str:
#     response = client.chat.completions.create(
#         model="llama3-8b-8192",   # free model on Groq
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content.strip()

# # def call_gemini(prompt):
# #     api_key = get_next_key()
# #     genai.configure(api_key=api_key)
# #     model = genai.GenerativeModel(MODEL_NAME)
# #     response = model.generate_content(prompt)
# #     return response.text.strip()

# def step_extract(raw_text):
#     prompt = f"""
# You are a support ticket parser.
# Extract these fields from the ticket:
# - customer: customer name or ID (if not found write "unknown")
# - product: product or service name (if not found write "unknown")
# - issue: one-line description of the problem

# Return ONLY valid JSON, no extra text, no markdown:
# {{"customer": "...", "product": "...", "issue": "..."}}

# Ticket:
# \"\"\"{raw_text}\"\"\"
# """
#     raw = call_gemini(prompt).strip()
#     if raw.startswith("```"):
#         raw = raw.split("```")[1]
#         if raw.startswith("json"):
#             raw = raw[4:]
#         raw = raw.strip()
#     return json.loads(raw)

# def step_classify(issue):
#     prompt = f"""
# Classify this support issue into exactly ONE category:
# - bug
# - feature_request
# - question

# Return ONLY the category word, nothing else.

# Issue: "{issue}"
# """
#     result = call_gemini(prompt).lower().strip()
#     valid = ["bug", "feature_request", "question"]
#     if result not in valid:
#         for v in valid:
#             if v in result:
#                 return v
#         return "question"
#     return result

# def step_summarize(raw_text, customer, product, category):
#     prompt = f"""
# Write a clear 1-2 sentence summary of this support ticket.
# Mention the customer, product, and type of issue.

# Customer: {customer}
# Product: {product}
# Category: {category}
# Ticket: \"\"\"{raw_text}\"\"\"

# Return ONLY the summary, no bullet points, no markdown.
# """
#     return call_gemini(prompt)

# def run_pipeline(raw_text, ticket_num):
#     print(f"\n{'─'*55}")
#     print(f"🎫 Ticket #{ticket_num}: {raw_text[:60]}...")

#     try:
#         print("   [Step 1] Extracting...")
#         extracted = step_extract(raw_text)
#         customer = extracted.get("customer", "unknown")
#         product  = extracted.get("product",  "unknown")
#         issue    = extracted.get("issue",    "unknown")
#         print(f"   ✔ customer={customer}, product={product}")
#         time.sleep(0.5)

#         print("   [Step 2] Classifying...")
#         category = step_classify(issue)
#         print(f"   ✔ category={category}")
#         time.sleep(2.0)

#         print("   [Step 3] Summarizing...")
#         summary = step_summarize(raw_text, customer, product, category)
#         print(f"   ✔ summary={summary[:60]}...")
#         time.sleep(0.5)

#         print("   [Step 4] Storing in DB...")
#         insert_ticket(raw_text, customer, product, issue, category, summary, status="success")
#         print("   ✔ Saved!")

#         return {"ticket_num": ticket_num, "status": "success", "category": category}

#     except Exception as e:
#         error_msg = str(e)
#         print(f"   ❌ FAILED: {error_msg}")
#         insert_failed_ticket(raw_text, fail_reason=error_msg)
#         return {"ticket_num": ticket_num, "status": "failed", "error": error_msg}
    
    
    #### HEY I AM USING GROK MODEL , HERE###

# pipeline.py
import json
import time
from groq import Groq
from config import GROQ_API_KEY
from db import insert_ticket, insert_failed_ticket

client = Groq(api_key=GROQ_API_KEY)

# ── Call Groq API ────────────────────────────────────────────────────────────

def call_groq(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# ── Step 1: Extract ──────────────────────────────────────────────────────────

def step_extract(raw_text):
    prompt = f"""
You are a support ticket parser.
Extract these fields from the ticket:
- customer: customer name or ID (if not found write "unknown")
- product: product or service name (if not found write "unknown")
- issue: one-line description of the problem

Return ONLY valid JSON, no extra text, no markdown:
{{"customer": "...", "product": "...", "issue": "..."}}

Ticket:
\"\"\"{raw_text}\"\"\"
"""
    raw = call_groq(prompt).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


# ── Step 2: Classify ─────────────────────────────────────────────────────────

def step_classify(issue):
    prompt = f"""
Classify this support issue into exactly ONE category:
- bug
- feature_request
- question

Return ONLY the category word, nothing else.

Issue: "{issue}"
"""
    result = call_groq(prompt).lower().strip()
    valid = ["bug", "feature_request", "question"]
    if result not in valid:
        for v in valid:
            if v in result:
                return v
        return "question"
    return result


# ── Step 3: Summarize ────────────────────────────────────────────────────────

def step_summarize(raw_text, customer, product, category):
    prompt = f"""
Write a clear 1-2 sentence summary of this support ticket.
Mention the customer, product, and type of issue.

Customer: {customer}
Product: {product}
Category: {category}
Ticket: \"\"\"{raw_text}\"\"\"

Return ONLY the summary text, no bullet points, no markdown.
"""
    return call_groq(prompt)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(raw_text, ticket_num):
    print(f"\n{'─'*55}")
    print(f"🎫 Ticket #{ticket_num}: {raw_text[:60]}...")

    try:
        print("   [Step 1] Extracting...")
        extracted = step_extract(raw_text)
        customer = extracted.get("customer", "unknown")
        product  = extracted.get("product",  "unknown")
        issue    = extracted.get("issue",    "unknown")
        print(f"   ✔ customer={customer}, product={product}")
        time.sleep(1)

        print("   [Step 2] Classifying...")
        category = step_classify(issue)
        print(f"   ✔ category={category}")
        time.sleep(1)

        print("   [Step 3] Summarizing...")
        summary = step_summarize(raw_text, customer, product, category)
        print(f"   ✔ summary={summary[:60]}...")
        time.sleep(1)

        print("   [Step 4] Storing in DB...")
        insert_ticket(raw_text, customer, product, issue, category, summary, status="success")
        print("   ✔ Saved!")

        return {"ticket_num": ticket_num, "status": "success", "category": category}

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ FAILED: {error_msg}")
        insert_failed_ticket(raw_text, fail_reason=error_msg)
        return {"ticket_num": ticket_num, "status": "failed", "error": error_msg}