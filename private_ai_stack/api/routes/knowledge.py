from fastapi import APIRouter, Depends

from private_ai_stack.api.dependencies import knowledge_service
from private_ai_stack.api.schemas import (
    KnowledgeDocumentRequest,
    KnowledgeDocumentResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)
from private_ai_stack.services.knowledge_service import KnowledgeService

router = APIRouter(tags=["knowledge"])


@router.post("/knowledge/documents", response_model=KnowledgeDocumentResponse)
async def ingest_document(
    payload: KnowledgeDocumentRequest, service: KnowledgeService = Depends(knowledge_service)
) -> KnowledgeDocumentResponse:
    return await service.ingest(payload)


@router.post("/knowledge/search", response_model=KnowledgeSearchResponse)
async def search(payload: KnowledgeSearchRequest, service: KnowledgeService = Depends(knowledge_service)) -> KnowledgeSearchResponse:
    return await service.search(payload)
