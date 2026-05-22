# AI CRM Assistant

Analyze sales leads using Google Gemini AI.
Returns a structured summary, follow-up action, and sentiment score.

## Setup

### 1. Clone & install
pip install -r requirements.txt

### 2. Configure environment
cp .env.example .env
# Add your Gemini API key inside .env

### 3. Run the server
uvicorn main:app --reload

### 4. API Docs
Visit: http://localhost:8000/docs

## Endpoint

POST /crm/analyze-lead
Content-Type: application/json

{
  "name": "Rahul Sharma",
  "company": "TechCorp India",
  "notes": "Very interested after demo. Wants pricing by Friday."
}

Response:
{
  "summary": "...",
  "suggestedFollowUp": "...",
  "sentimentScore": "positive"
}

## Run Tests
pytest tests/test_crm.py -v