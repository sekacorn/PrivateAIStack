"""Rebuilding a vector store with a different embedding model, offline.

Phase 3a of the vendor-exit work asks whether a store *can* be rebuilt. This
package does the rebuilding: it reads an ``embedding-portability.v1`` document
with retained source text, re-embeds it with a different model in resumable
batches, and writes both the rebuilt store and a ``re-embed-record.v1``
describing the run.

Nothing here opens a socket or loads weights. Remote providers arrive in 3b-5
behind an explicit opt-in, and the retrieval-quality comparison in 3b-4, so a
record written today carries no quality claim at all rather than an empty one.
"""

from private_ai_stack.reembed.checkpoint import Checkpoint, CheckpointError, load_checkpoint
from private_ai_stack.reembed.embedders import (
    AGENTFORGE_HASHED_BOW,
    BUILT_IN_MODEL_IDS,
    PRIVATEAISTACK_HASHED_BOW,
    Embedder,
    EmbedderError,
    ModelIdentity,
    build_built_in_embedder,
)
from private_ai_stack.reembed.evidence import EvidenceError, SourceRecord, SourceStore, load_source_store
from private_ai_stack.reembed.pipeline import DEFAULT_BATCH_SIZE, ReEmbedError, ReEmbedOutcome, re_embed_store

__all__ = [
    "AGENTFORGE_HASHED_BOW",
    "BUILT_IN_MODEL_IDS",
    "DEFAULT_BATCH_SIZE",
    "PRIVATEAISTACK_HASHED_BOW",
    "Checkpoint",
    "CheckpointError",
    "Embedder",
    "EmbedderError",
    "EvidenceError",
    "ModelIdentity",
    "ReEmbedError",
    "ReEmbedOutcome",
    "SourceRecord",
    "SourceStore",
    "build_built_in_embedder",
    "load_checkpoint",
    "load_source_store",
    "re_embed_store",
]
