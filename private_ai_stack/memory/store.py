import hashlib
import json
from typing import Any

from private_ai_stack.config.settings import Settings
from private_ai_stack.memory.embeddings import DeterministicEmbeddingModel, cosine_similarity
from private_ai_stack.memory.models import DocumentChunk


class MemoryStore:
    """Stores document chunks in PostgreSQL when available, with an in-memory fallback for tests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_model = DeterministicEmbeddingModel(settings.embedding_dimensions)
        self._chunks: list[DocumentChunk] = []
        self._loaded = False
        self._connection: Any | None = None

    async def close(self) -> None:
        if self._connection is not None:
            self._connection.close()

    async def status(self) -> str:
        try:
            await self._ensure_loaded()
        except Exception:
            return "unavailable"
        return "ok"

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.settings.database_url.startswith("postgresql"):
            try:
                import psycopg

                self._connection = psycopg.connect(self.settings.database_url, connect_timeout=3)
                with self._connection.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS documents (
                            document_id TEXT PRIMARY KEY,
                            source_name TEXT NOT NULL,
                            content_hash TEXT NOT NULL,
                            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS document_chunks (
                            chunk_id TEXT PRIMARY KEY,
                            document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                            source_name TEXT NOT NULL,
                            text TEXT NOT NULL,
                            content_hash TEXT NOT NULL,
                            embedding JSONB NOT NULL,
                            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    )
                    self._connection.commit()
            except Exception:
                self._connection = None
                raise
        self._loaded = True

    def chunk_text(self, content: str, size: int = 1200, overlap: int = 120) -> list[str]:
        """Split content into overlapping chunks so local search keeps nearby context together."""
        chunks: list[str] = []
        start = 0
        while start < len(content):
            chunk = content[start : start + size].strip()
            if chunk:
                chunks.append(chunk)
            start += max(1, size - overlap)
        return chunks

    async def ingest(
        self, content: str, source_name: str, metadata: dict[str, Any], replace_existing: bool = False
    ) -> tuple[str, int, bool, str]:
        await self._ensure_loaded()
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        document_id = hashlib.sha256(f"{source_name}:{content_hash}".encode()).hexdigest()[:24]
        chunks = self.chunk_text(content)
        records = [
            DocumentChunk(
                document_id=document_id,
                chunk_id=f"{document_id}-{index:04d}",
                source_name=source_name,
                text=chunk,
                content_hash=content_hash,
                embedding=self.embedding_model.embed(chunk),
                metadata=metadata,
            )
            for index, chunk in enumerate(chunks)
        ]

        # The in-memory branch keeps unit tests and local fallback mode free of PostgreSQL dependencies.
        if self._connection is None:
            existing = [chunk for chunk in self._chunks if chunk.document_id == document_id]
            if existing and not replace_existing:
                return document_id, 0, True, content_hash
            self._chunks = [chunk for chunk in self._chunks if chunk.document_id != document_id]
            self._chunks.extend(records)
            return document_id, len(records), False, content_hash

        with self._connection.cursor() as cur:
            cur.execute("SELECT document_id FROM documents WHERE document_id = %s", (document_id,))
            exists = cur.fetchone() is not None
            if exists and not replace_existing:
                return document_id, 0, True, content_hash
            cur.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))
            cur.execute(
                "INSERT INTO documents (document_id, source_name, content_hash, metadata) VALUES (%s, %s, %s, %s)",
                (document_id, source_name, content_hash, json.dumps(metadata)),
            )
            for record in records:
                cur.execute(
                    """
                    INSERT INTO document_chunks (chunk_id, document_id, source_name, text, content_hash, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.chunk_id,
                        record.document_id,
                        record.source_name,
                        record.text,
                        record.content_hash,
                        json.dumps(record.embedding),
                        json.dumps(record.metadata),
                    ),
                )
            self._connection.commit()
        return document_id, len(records), False, content_hash

    async def search(self, query: str, limit: int = 5) -> list[tuple[DocumentChunk, float]]:
        await self._ensure_loaded()
        query_embedding = self.embedding_model.embed(query)
        chunks = await self._all_chunks()
        # v0.1 favors deterministic, portable scoring over a production ANN index.
        scored = [(chunk, cosine_similarity(query_embedding, chunk.embedding)) for chunk in chunks]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    async def _all_chunks(self) -> list[DocumentChunk]:
        if self._connection is None:
            return list(self._chunks)
        with self._connection.cursor() as cur:
            cur.execute(
                "SELECT document_id, chunk_id, source_name, text, content_hash, embedding, metadata, created_at FROM document_chunks"
            )
            rows = cur.fetchall()
        return [
            DocumentChunk(
                document_id=row[0],
                chunk_id=row[1],
                source_name=row[2],
                text=row[3],
                content_hash=row[4],
                embedding=[float(value) for value in row[5]],
                metadata=dict(row[6]),
                created_at=row[7],
            )
            for row in rows
        ]

    async def export_jsonl(self, path: str) -> str:
        chunks = await self._all_chunks()
        with open(path, "w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")
        return path
