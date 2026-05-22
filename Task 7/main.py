from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from models import LeadRequest, LeadResponse
from llm_service import analyze_lead

app = FastAPI(
    title="AI CRM Assistant",
    description="Analyze sales leads using Gemini AI — summary, follow-up, and sentiment.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"status": "AI CRM Assistant is running"}


@app.post("/crm/analyze-lead", response_model=LeadResponse)
def analyze_lead_endpoint(lead: LeadRequest):
    try:
        result = analyze_lead(
            name=lead.name,
            company=lead.company,
            notes=lead.notes,
        )
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    
@app.post("/crm/debug")
def debug_endpoint(lead: LeadRequest):
    try:
        result = analyze_lead(
            name=lead.name,
            company=lead.company,
            notes=lead.notes,
        )
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e), "type": type(e).__name__}