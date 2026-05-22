def build_crm_prompt(name: str, company: str, notes: str) -> str:
    return f"""
You are an expert CRM AI assistant helping sales teams analyze leads.

Analyze the following lead information and return ONLY a valid JSON object.
Do NOT include any explanation, markdown, or extra text — just raw JSON.

Return exactly this structure:
{{
  "summary": "2-3 sentence summary of the lead situation",
  "suggestedFollowUp": "One specific, actionable next step the sales rep should take",
  "sentimentScore": "positive" or "neutral" or "negative"
}}

Rules:
- summary: Must be 2-3 sentences. Capture key intent, concerns, and status.
- suggestedFollowUp: Must be one concrete action (e.g., "Send pricing proposal by Friday").
- sentimentScore: Must be exactly one of: positive, neutral, negative.

Lead Information:
- Name: {name}
- Company: {company}
- Notes: {notes}
""".strip()