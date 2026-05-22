import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from models import EmailRequest, EmailResponse, ErrorResponse
from analyzer import analyze_email

load_dotenv()

# ─── App Setup ─────────────────────────────────────────────
app = FastAPI(
    title="AI Email Reply Generator API",
    description="Analyzes emails and returns tone, summary, and a suggested reply using Gemini AI.",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc"      # ReDoc UI at /redoc
)


# ─── Health Check ──────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "message": "AI Email Reply Generator API is live.",
        "usage": "POST /email/analyze with body { 'email': '...' }"
    }


@app.get("/health", tags=["Health"])
def health_check():
    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    return {
        "status": "ok",
        "gemini_key_configured": api_key_set
    }


# ─── Main Endpoint ─────────────────────────────────────────
@app.post(
    "/email/analyze",
    response_model=EmailResponse,
    tags=["Email Analysis"],
    summary="Analyze an email and get tone, summary, and suggested reply",
    responses={
        200: {"description": "Successful analysis"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        413: {"model": ErrorResponse, "description": "Email too long"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)
async def analyze_email_endpoint(request: EmailRequest):
    """
    Analyze an email and return:
    - **tone**: formal | neutral | urgent | casual
    - **summary**: one-sentence description
    - **suggestedReply**: ready-to-send reply
    - **debug**: token usage, latency, and cost
    """

    email_text = request.email.strip()

    # ─── Guard: Empty After Strip ──────────────────────────
    if not email_text:
        raise HTTPException(
            status_code=400,
            detail="Email body cannot be empty or whitespace only."
        )

    # ─── Guard: Length Limit (8000 chars) ─────────────────
    if len(email_text) > 8000:
        raise HTTPException(
            status_code=413,
            detail=f"Email exceeds maximum length of 8000 characters. "
                   f"Received: {len(email_text)} characters."
        )

    # ─── Call Analyzer ─────────────────────────────────────
    try:
        result = analyze_email(email_text)
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse AI response: {str(e)}"
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI service error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


# ─── Global Exception Handler ──────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )