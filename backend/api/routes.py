from fastapi import APIRouter
from ..services.generator_service import GeneratorService
from .schemas import QueryRequest, QueryResponse

router = APIRouter()
service = GeneratorService()

@router.post("/generate", response_model=QueryResponse)
def generate(request: QueryRequest):
    result = service.generate(request.query)
    return QueryResponse(response=result)