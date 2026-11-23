"""The built-in, deterministic embedders a re-embedding run may target.

Phase 3b rebuilds a vector store with a *different* model and asks whether
retrieval still behaves acceptably. Checkpoint 3b-2 deliberately restricts the
target to the two hashing schemes this ecosystem already ships, because they
are pure arithmetic over :mod:`hashlib`: no weights, no sockets, no hardware
variance. A run against them is reproducible by anyone holding the source text,
which is what lets a later checkpoint re-run it rather than take its word.

Model identity here is the triple ``id`` + ``revision`` + ``dimensions``. The
same scheme at two dimensions is two models, because the bucket a token lands
in is ``hash(token) % dimensions``: change the dimension and every component
moves. No backend default is ever consulted, and a declared dimension that the
produced vector does not have is an error rather than something to paper over.

``revision`` is the only part of an identity that can carry a change of
behaviour, so **any change to an embedder's output requires a new revision**.
Otherwise a refactor silently invalidates every store and record produced under
the old behaviour while the identity in the evidence still reads the same.
``tests/fixtures/reembed/golden-vectors.json`` pins the current output of every
identity below and fails the moment one moves.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

MAX_DIMENSIONS: Final = 8192
"""Matches ``embedding-portability.v1``, so an identity valid here is valid there."""

AGENTFORGE_HASHED_BOW: Final = "agentforge.hashed-bow.blake2b"
PRIVATEAISTACK_HASHED_BOW: Final = "privateaistack.hashed-bow.sha256-signed"


class EmbedderError(ValueError):
    """Raised when a requested embedder cannot be built as declared."""


@dataclass(frozen=True)
class ModelIdentity:
    """What a vector was produced by, as far as a reader can check it.

    ``availability``, a weights digest and a license are operator declarations
    that no offline scanner can re-run, so they live in the evidence document
    rather than here. This triple is the part that determines the arithmetic.
    """

    id: str
    revision: str
    dimensions: int

    def __post_init__(self) -> None:
        if not self.id or len(self.id) > 128:
            raise EmbedderError("Model id must be 1 to 128 characters.")
        if not self.revision or len(self.revision) > 128:
            raise EmbedderError("Model revision must be 1 to 128 characters.")
        if not 1 <= self.dimensions <= MAX_DIMENSIONS:
            raise EmbedderError(f"Model dimensions must be between 1 and {MAX_DIMENSIONS}.")

    @property
    def triple(self) -> tuple[str, str, int]:
        return (self.id, self.revision, self.dimensions)


@runtime_checkable
class Embedder(Protocol):
    """The whole seam a re-embedding run needs.

    Narrow on purpose: a pipeline that can only ask for one vector at a time
    cannot accidentally hold a whole store in memory, and a replacement
    embedder has one method to implement.
    """

    @property
    def identity(self) -> ModelIdentity: ...

    def embed(self, text: str) -> list[float]: ...


class AgentForgeHashedBagOfWords:
    """A compatible reimplementation of AgentForge's in-memory store embedding.

    Unsigned term frequency over ``blake2b`` buckets, L2-normalised. The
    original lives in ``forge.memory.vector.InMemoryVectorStore`` as a private
    method on a class whose dimension is also private, so re-embedding calls
    this instead of reaching into it. ``tests/unit/test_reembed_embedders.py``
    holds the two to the same output through AgentForge's public search scores,
    so "compatible" is a tested claim and not a comment.
    """

    _TOKEN: Final = re.compile(r"[a-z0-9]+")

    def __init__(self, dimensions: int) -> None:
        self._identity = ModelIdentity(id=AGENTFORGE_HASHED_BOW, revision="1", dimensions=dimensions)

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def embed(self, text: str) -> list[float]:
        dimensions = self._identity.dimensions
        values = [0.0] * dimensions
        for token in self._TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            values[int.from_bytes(digest, "big") % dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in values))
        if norm > 0:
            values = [value / norm for value in values]
        return values


class PrivateAIStackSignedHashedBagOfWords:
    """The scheme :class:`private_ai_stack.memory.embeddings.DeterministicEmbeddingModel` uses.

    Signed unit weights over ``sha256`` buckets, L2-normalised, tokenised by
    whitespace rather than by a character class. It is reimplemented here for
    the same reason as the AgentForge scheme, with one substantive difference:
    the memory store's version divides by ``norm or 1.0``, so an all-zero
    vector stays all-zero either way, and the two agree component for
    component. ``tests/unit/test_reembed_embedders.py`` asserts that directly
    against the shipped model.
    """

    def __init__(self, dimensions: int) -> None:
        self._identity = ModelIdentity(id=PRIVATEAISTACK_HASHED_BOW, revision="1", dimensions=dimensions)

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def embed(self, text: str) -> list[float]:
        dimensions = self._identity.dimensions
        values = [0.0] * dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            values[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


_FACTORIES: Final[dict[str, tuple[str, Callable[[int], Embedder]]]] = {
    AGENTFORGE_HASHED_BOW: ("1", AgentForgeHashedBagOfWords),
    PRIVATEAISTACK_HASHED_BOW: ("1", PrivateAIStackSignedHashedBagOfWords),
}

BUILT_IN_MODEL_IDS: Final = tuple(sorted(_FACTORIES))


def build_built_in_embedder(identity: ModelIdentity) -> Embedder:
    """Return the built-in embedder matching ``identity`` exactly, or refuse.

    Refusing is the point. An unknown id, or a known id at a revision this
    release does not implement, would otherwise produce vectors under a name
    that promises something else, and nothing downstream could tell.
    """
    try:
        revision, factory = _FACTORIES[identity.id]
    except KeyError:
        raise EmbedderError(f"Model id {identity.id!r} is not a built-in embedder. Available: {', '.join(BUILT_IN_MODEL_IDS)}.") from None
    if identity.revision != revision:
        raise EmbedderError(f"Built-in embedder {identity.id!r} is revision {revision!r}, not {identity.revision!r}.")
    return factory(identity.dimensions)
