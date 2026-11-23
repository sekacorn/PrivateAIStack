"""Builders for the ``re-embed-record.v1`` documents this package actually emits.

The contract test and the committed fixtures both come from here, so the shape
checked against the AI Infrastructure Suite is the shape a real run writes
rather than a hand-maintained copy of it. A field added to the egress block
shows up in both places without anyone remembering to update a literal.

Every variant is produced by running the real pipeline over a store built under
a temporary directory. Nothing here opens a socket: the external executor is a
registry-owned embedder standing in for operator code, which is what lets the
external and hosted shapes be produced offline.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from private_ai_stack.reembed import (
    EXTERNAL_UNVERIFIED,
    HOSTED,
    ModelIdentity,
    provider_data_preflight,
    re_embed_store,
)
from private_ai_stack.reembed.embedders import build_built_in_embedder
from tests.unit.reembed_support import build_source_store

MODEL: Final = ModelIdentity(id="privateaistack.hashed-bow.sha256-signed", revision="1", dimensions=8)

#: Every record variant this package can write, by fixture name. The contract
#: test runs all of them, so a new variant cannot be added without being
#: validated against the suite.
VARIANTS: Final = ("local", "external-local", "hosted")


def _texts() -> dict[str, str]:
    return {
        "procedure-escalation": "Escalate to the on-call engineer when the queue depth exceeds 40.",
        "procedure-password-reset": "Reset the password through the self-service portal.",
        "procedure-refund": "Refunds over 500 need a supervisor approval and an audit entry.",
    }


def _run(root: Path, name: str, **kwargs: Any) -> Path:
    source = build_source_store(
        root / f"{name}-evidence",
        texts=_texts(),
        missing=("gone",),
        deleted_by_policy=("purged",),
        dimensions=8,
    )
    outcome = re_embed_store(
        source_document=source,
        source_store_id="support-memory",
        target_directory=root / f"{name}-target",
        target_store_id="support-memory-rebuilt",
        target_model=MODEL,
        availability="open_weight_self_hostable",
        **kwargs,
    )
    return outcome.record


def build_local_record(root: Path) -> Path:
    """A registry-owned run. Nothing is handed to anything."""
    return _run(root, "local")


def build_external_local_record(root: Path) -> Path:
    """An injected executor on a loopback endpoint. Nothing leaves the machine."""
    gate = provider_data_preflight(
        executor_kind=EXTERNAL_UNVERIFIED,
        provider="local-test-executor",
        endpoint="http://127.0.0.1:11434/embed",
        allow_external=True,
    )
    return _run(root, "external-local", egress=gate, embedder=build_built_in_embedder(MODEL))


def build_hosted_record(root: Path) -> Path:
    """An injected executor the operator allowed to send content off the machine."""
    gate = provider_data_preflight(
        executor_kind=EXTERNAL_UNVERIFIED,
        provider="example-hosted",
        endpoint="https://embeddings.example/v1",
        hosting_mode=HOSTED,
        dataset_privacy_classification="public",
        allow_external=True,
        allow_hosted=True,
        telemetry_enabled=True,
    )
    return _run(root, "hosted", egress=gate, embedder=build_built_in_embedder(MODEL))


BUILDERS: Final[dict[str, Callable[[Path], Path]]] = {
    "local": build_local_record,
    "external-local": build_external_local_record,
    "hosted": build_hosted_record,
}


def build_all(root: Path) -> dict[str, dict[str, Any]]:
    """Produce every variant as a parsed document, keyed by fixture name."""
    produced: dict[str, dict[str, Any]] = {}
    for name, builder in BUILDERS.items():
        path = builder(root)
        produced[name] = json.loads(path.read_text(encoding="utf-8"))
    return produced
