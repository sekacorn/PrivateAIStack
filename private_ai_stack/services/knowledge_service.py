from private_ai_stack.api.errors import AppError
from private_ai_stack.api.schemas import (
    KnowledgeDocumentRequest,
    KnowledgeDocumentResponse,
    KnowledgeHit,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)
from private_ai_stack.audit.writer import AuditWriter
from private_ai_stack.memory.embeddings import EmbeddingDimensionError
from private_ai_stack.memory.store import MemoryStore, PersistenceUnavailable


class KnowledgeService:
    def __init__(self, memory: MemoryStore, audit: AuditWriter) -> None:
        self.memory = memory
        self.audit = audit

    async def ingest(self, payload: KnowledgeDocumentRequest) -> KnowledgeDocumentResponse:
        try:
            document_id, count, idempotent, content_hash = await self.memory.ingest(
                payload.content,
                payload.source_name,
                payload.metadata,
                payload.replace_existing,
            )
        except ValueError as exc:
            raise AppError("knowledge_input_rejected", "Knowledge document exceeds configured local limits.", 413) from exc
        except PersistenceUnavailable as exc:
            raise AppError("persistence_unavailable", "Configured knowledge persistence is unavailable.", 503) from exc
        self.audit.write(
            "knowledge.ingested",
            entity_type="document",
            entity_id=document_id,
            details={"source_name": payload.source_name, "chunks_created": count, "idempotent": idempotent, "content_hash": content_hash},
        )
        return KnowledgeDocumentResponse(document_id=document_id, chunks_created=count, idempotent=idempotent, content_hash=content_hash)

    async def search(self, payload: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        try:
            results = await self.memory.search(payload.query, payload.limit)
        except PersistenceUnavailable as exc:
            raise AppError("persistence_unavailable", "Configured knowledge persistence is unavailable.", 503) from exc
        except EmbeddingDimensionError as exc:
            raise AppError(
                "knowledge_index_inconsistent",
                "Stored knowledge vectors and the configured embedding dimension differ. "
                "Restore the configured dimension or rebuild the index; search results would be wrong.",
                409,
            ) from exc
        hits = [
            KnowledgeHit(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                source_name=chunk.source_name,
                score=score,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for chunk, score in results
        ]
        self.audit.write(
            "knowledge.searched",
            entity_type="knowledge",
            entity_id="search",
            details={"query_length": len(payload.query), "hits": len(hits)},
        )
        return KnowledgeSearchResponse(query=payload.query, hits=hits)
