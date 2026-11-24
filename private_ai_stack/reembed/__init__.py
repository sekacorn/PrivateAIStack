"""Rebuilding a vector store with a different embedding model, offline.

Phase 3a of the vendor-exit work asks whether a store *can* be rebuilt. This
package does the rebuilding: it reads an ``embedding-portability.v1`` document
with retained source text, re-embeds it with a different model in resumable
batches, and writes both the rebuilt store and a ``re-embed-record.v1``
describing the run.

By default a run uses a registry-owned local embedder. An injected embedder is
``external_unverified`` and is refused before source files are read unless an
operator explicitly allows it. The package records what it submitted without
claiming OS-level network isolation or verified behaviour for arbitrary code.

:func:`compare_stores` answers the question that follows the rebuild: do the
two stores still put the same records near each other? It reports neighborhood
agreement, never search quality, and a comparison over an operator-chosen query
set is reported in full and moves no lock-in score.

Before committing to a run an operator can ask what it will cost:
:func:`estimate_re_embed` embeds a small calibration sample, measures it, and
extrapolates to the whole store as a ``re-embed-estimate.v1`` forecast. The
estimate is not evidence and is deliberately not reproducible, because measured
latency depends on the machine.
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
from private_ai_stack.reembed.estimate import (
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
    Calibration,
    EstimateError,
    ReEmbedEstimate,
    estimate_re_embed,
    load_cost_profile,
)
from private_ai_stack.reembed.evidence import EvidenceError, SourceRecord, SourceStore, load_source_store
from private_ai_stack.reembed.pipeline import DEFAULT_BATCH_SIZE, ReEmbedError, ReEmbedOutcome, re_embed_store
from private_ai_stack.reembed.privacy import (
    DEFAULT_CLASSIFICATION,
    EXECUTOR_KINDS,
    EXTERNAL_UNVERIFIED,
    HANDED_TO_EXECUTOR,
    HOSTED,
    HOSTING_MODES,
    LOCAL,
    LOCAL_BUILTIN,
    NO_TRANSMISSION,
    PRIVACY_CLASSIFICATIONS,
    TRANSMISSION_BASES,
    EgressError,
    EgressPreflight,
    local_egress,
    provider_data_preflight,
)
from private_ai_stack.reembed.quality import (
    DEFAULT_ANCHOR_COUNT,
    DEFAULT_CANDIDATE_POOL,
    DEFAULT_MINIMUM_SAMPLE_SIZE,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    QUERY_SET_SCHEMA_VERSION,
    SEEDED_RANDOM,
    UNJUDGED,
    QualityError,
    QualityResult,
    QualityThresholds,
    QuerySet,
    VectorStore,
    attach_quality,
    compare_stores,
    derive_anchor_seed,
    draw_anchors,
    load_query_set,
    load_vector_store,
)

__all__ = [
    "AGENTFORGE_HASHED_BOW",
    "BUILT_IN_MODEL_IDS",
    "DEFAULT_ANCHOR_COUNT",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CLASSIFICATION",
    "HANDED_TO_EXECUTOR",
    "NO_TRANSMISSION",
    "TRANSMISSION_BASES",
    "DEFAULT_CANDIDATE_POOL",
    "DEFAULT_MINIMUM_SAMPLE_SIZE",
    "DEFAULT_SAMPLE_SEED",
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_SEED",
    "DEFAULT_TOP_K",
    "EXECUTOR_KINDS",
    "EXTERNAL_UNVERIFIED",
    "HOSTED",
    "HOSTING_MODES",
    "LOCAL",
    "LOCAL_BUILTIN",
    "PRIVACY_CLASSIFICATIONS",
    "PRIVATEAISTACK_HASHED_BOW",
    "QUERY_SET_SCHEMA_VERSION",
    "SEEDED_RANDOM",
    "UNJUDGED",
    "Calibration",
    "Checkpoint",
    "CheckpointError",
    "Embedder",
    "EgressError",
    "EgressPreflight",
    "EmbedderError",
    "EstimateError",
    "EvidenceError",
    "ModelIdentity",
    "QualityError",
    "QualityResult",
    "QualityThresholds",
    "QuerySet",
    "ReEmbedError",
    "ReEmbedEstimate",
    "ReEmbedOutcome",
    "SourceRecord",
    "SourceStore",
    "VectorStore",
    "attach_quality",
    "build_built_in_embedder",
    "compare_stores",
    "derive_anchor_seed",
    "draw_anchors",
    "estimate_re_embed",
    "load_checkpoint",
    "load_cost_profile",
    "load_query_set",
    "load_source_store",
    "local_egress",
    "load_vector_store",
    "provider_data_preflight",
    "re_embed_store",
]
