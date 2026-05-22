import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ── Mock responses
MOCK_POSITIVE = {
    "summary": "The lead is highly interested and ready to move forward after the demo.",
    "suggestedFollowUp": "Send the enterprise contract and pricing details within 24 hours.",
    "sentimentScore": "positive"
}

MOCK_NEGATIVE = {
    "summary": "The lead is unresponsive and shows no interest in the product.",
    "suggestedFollowUp": "Send a final follow-up email and mark the lead as cold.",
    "sentimentScore": "negative"
}

MOCK_NEUTRAL = {
    "summary": "The lead is undecided and needs more time before making a decision.",
    "suggestedFollowUp": "Schedule a follow-up call in 2 weeks to revisit their needs.",
    "sentimentScore": "neutral"
}


def post_lead(name, company, notes):
    return client.post("/crm/analyze-lead", json={
        "name": name,
        "company": company,
        "notes": notes
    })


def assert_valid_response(res):
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "suggestedFollowUp" in data
    assert "sentimentScore" in data
    assert data["sentimentScore"] in ("positive", "neutral", "negative")
    assert len(data["summary"]) > 10
    assert len(data["suggestedFollowUp"]) > 5


# ── Test 1: Positive interested lead
@patch("main.analyze_lead", return_value=MOCK_POSITIVE)
def test_positive_interested_lead(mock_llm):
    res = post_lead(
        "Rahul Sharma",
        "TechCorp India",
        "Called twice, very excited about our enterprise plan. Loved the demo. Ready to sign next week."
    )
    assert_valid_response(res)
    assert res.json()["sentimentScore"] == "positive"


# ── Test 2: Cold unresponsive lead
@patch("main.analyze_lead", return_value=MOCK_NEGATIVE)
def test_cold_unresponsive_lead(mock_llm):
    res = post_lead(
        "Priya Mehta",
        "Coldstone Ltd",
        "No response to 3 emails and 2 calls. Left voicemail. Seems completely uninterested."
    )
    assert_valid_response(res)
    assert res.json()["sentimentScore"] == "negative"


# ── Test 3: Neutral lead
@patch("main.analyze_lead", return_value=MOCK_NEUTRAL)
def test_neutral_lead(mock_llm):
    res = post_lead(
        "Amit Verma",
        "MidLine Services",
        "Had a short call. Not sure if they need the product now. Will revisit in Q3."
    )
    assert_valid_response(res)
    assert res.json()["sentimentScore"] == "neutral"


# ── Test 4: Very short notes
@patch("main.analyze_lead", return_value=MOCK_NEUTRAL)
def test_short_notes(mock_llm):
    res = post_lead("John Doe", "Acme Corp", "Interested.")
    assert_valid_response(res)


# ── Test 5: Very long notes
@patch("main.analyze_lead", return_value=MOCK_POSITIVE)
def test_long_notes(mock_llm):
    res = post_lead(
        "Sarah Connor",
        "Future Systems",
        ("Had a 45-minute product demo. Sarah asked detailed questions about API integrations, "
         "data security, GDPR compliance, SLA guarantees, and onboarding timelines. "
         "She mentioned budget approval is pending with her CFO. Team size is 200+. "
         "Competing with two other vendors. Follow-up scheduled for next Tuesday. "
         "She seemed very positive about the UI and impressed by the dashboard analytics. "
         "Price is a concern she wants a 20% discount for annual commitment.")
    )
    assert_valid_response(res)


# ── Test 6: Missing name field → 422
def test_missing_name_field():
    res = client.post("/crm/analyze-lead", json={
        "company": "SomeCorp",
        "notes": "Interested in product."
    })
    assert res.status_code == 422


# ── Test 7: Missing company field → 422
def test_missing_company_field():
    res = client.post("/crm/analyze-lead", json={
        "name": "John",
        "notes": "Interested in product."
    })
    assert res.status_code == 422


# ── Test 8: Hindi mixed notes
@patch("main.analyze_lead", return_value=MOCK_POSITIVE)
def test_hindi_mixed_notes(mock_llm):
    res = post_lead(
        "Vikram Singh",
        "Desi Tech",
        "Bahut interested lag rahe hain. Demo ke baad unhone pricing poochi. Budget thoda tight hai but keen hain."
    )
    assert_valid_response(res)


# ── Test 9: Aggressive rude lead
@patch("main.analyze_lead", return_value=MOCK_NEGATIVE)
def test_aggressive_lead_tone(mock_llm):
    res = post_lead(
        "Angry Andy",
        "Rage Corp",
        "Called and was very rude. Said our product is overpriced junk. Threatened to leave a bad review online."
    )
    assert_valid_response(res)
    assert res.json()["sentimentScore"] == "negative"


# ── Test 10: Already converted lead
@patch("main.analyze_lead", return_value=MOCK_POSITIVE)
def test_already_converted_lead(mock_llm):
    res = post_lead(
        "Lisa Ray",
        "WonDeal Inc",
        "Deal closed yesterday. Signed the annual enterprise contract. Onboarding starts Monday."
    )
    assert_valid_response(res)
    assert res.json()["sentimentScore"] == "positive"


# ── Test 11: Refund cancellation lead
@patch("main.analyze_lead", return_value=MOCK_NEGATIVE)
def test_refund_cancellation_lead(mock_llm):
    res = post_lead(
        "Tom Hardy",
        "Refund Ltd",
        "Wants to cancel subscription. Not satisfied with support response times. Requesting full refund."
    )
    assert_valid_response(res)
    assert res.json()["sentimentScore"] == "negative"


# ── Test 12: Invalid JSON body → 422
def test_invalid_json_body():
    res = client.post(
        "/crm/analyze-lead",
        content="not-json",
        headers={"Content-Type": "application/json"}
    )
    assert res.status_code == 422


# ── Test 13: Empty notes → 422
def test_empty_notes():
    res = client.post("/crm/analyze-lead", json={
        "name": "Empty Note",
        "company": "NoNote Corp",
        "notes": ""
    })
    assert res.status_code == 422