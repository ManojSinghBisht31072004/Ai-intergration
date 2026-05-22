import json
import logging
from dataclasses import datacalss,field
from pydantic import ValidationError
from schema import RAGResponse

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    response: RAGResponse | None = None
    errors : list[str] = field(deafult_factor=list)
    raw_json : dict | None = None
    

def validate_rag_response(raw: str | dict) -> ValidationResult:
    
