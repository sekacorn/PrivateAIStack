# Re-embedding a vector store

Phase 3a of the vendor-exit work asks whether a store *can* be rebuilt.
`private_ai_stack.reembed` does the rebuilding: it reads an
`embedding-portability.v1` document with retained source text, re-embeds it
with a different model in resumable batches, and writes both the rebuilt store
and a `re-embed-record.v1` describing the run.

The full design, including the checkpoints after this one, is in the umbrella
repository at `docs/phase3b-re-embedding.md`.

## What a run produces

Given a source evidence document and a target directory:

| Path | What it is |
|---|---|
| `vectors/shard-NNNN.jsonl` | the rebuilt vectors, one shard per batch |
| `sources/record-NNNN.txt` | the store's own copy of the text it was built from |
| `embedding-portability.json` | the rebuilt store, as an ordinary `embedding-portability.v1` document |
| `re-embed-record.json` | a `re-embed-record.v1`: what was rebuilt, what was skipped, how it was batched, what left the machine |
| `checkpoint.json` | resume state (`re-embed-checkpoint.v1`), not evidence and never read by `exit-report` |

The rebuilt store is an ordinary 3a input, so `ai-suite exit-report` scores it
with no new code.

```python
from pathlib import Path

from private_ai_stack.reembed import ModelIdentity, re_embed_store

outcome = re_embed_store(
    source_document=Path("evidence/embedding-portability.json"),
    source_store_id="support-memory",
    target_directory=Path("rebuilt"),
    target_store_id="support-memory-rebuilt",
    target_model=ModelIdentity(id="privateaistack.hashed-bow.sha256-signed", revision="1", dimensions=64),
    availability="open_weight_self_hostable",
)
print(outcome.retained_rebuilt, "records rebuilt in", outcome.batches_completed, "batches")
```

`availability` has no default. It is an operator attestation that no offline
check can confirm, and defaulting it would put a claim into the evidence that
nobody made.

## Model identity is a triple

`id` + `revision` + `dimensions`. The same scheme at two dimensions is two
models, because the bucket a token lands in is `hash(token) % dimensions`:
change the dimension and every component moves. No backend default is ever
consulted, and an embedder that produces a width other than the declared one
stops the run before anything is written.

Two built-in embedders are available, both pure arithmetic over `hashlib`:

| `id` | Scheme |
|---|---|
| `agentforge.hashed-bow.blake2b` | unsigned term frequency over `blake2b` buckets |
| `privateaistack.hashed-bow.sha256-signed` | signed unit weights over `sha256` buckets |

Each is a compatible reimplementation of code this ecosystem already ships, and
the tests hold them to the originals' output rather than asserting it in a
comment. An unknown model id is refused rather than approximated.

## Changing an embedder requires a new revision

An identity is a promise about arithmetic, and `revision` is the only part of
it that can carry a change. So:

> **Any change to an embedder's output requires a new `revision` in its model
> identity.** Refactoring that provably leaves every vector untouched does not.

Without that rule a refactor silently invalidates every store, record and
digest already produced under the old behaviour, and no reader can tell: the
identity in the evidence still reads the same, and the evidence still verifies
against itself. A 3b-6 scanner re-running a rebuild would then get different
vectors from a record that looks correct, and the honest conclusion available
to it — that the operator is wrong — would be the wrong one.

`tests/fixtures/reembed/golden-vectors.json` pins the exact output of every
built-in identity, at more than one width because the bucket a token lands in
depends on the width. `tests/unit/test_reembed_golden_vectors.py` fails if any
of it moves. When it fails, bump the revision; editing the fixture to match new
output is the one response that is never correct.

Conformance against AgentForge is checked twice: in the ordinary test run
against the installed `agentforge-oss`, and in CI's `embedder-conformance` job
against the latest release in the range this project declares, so a newer
release that changed the scheme cannot pass unnoticed behind a pinned floor.

## Interruption and resume

A run over a large store is long and interruptible, so it is a resumable batch
job. The ordering rule is the whole of the crash safety story: the batch's
shard reaches disk first, and only then is the checkpoint updated to claim it,
atomically. So a crash can leave a shard that nothing claims, but never a claim
without its shard.

- An **unclaimed shard** is an unfinished batch. It is discarded and
  re-embedded, because nothing on disk says whether it is complete.
- A **claimed shard whose bytes have changed** was altered by something other
  than the run. That is an error, not a reason to quietly redo the work.
- A checkpoint written against a **different input document, target, model
  triple or batch size** is refused. Each of those changes which vectors belong
  where.
- Each completed batch must hold exactly the records this run would have put in
  that batch, in that order. The batching is recomputed and compared rather
  than trusted, so an edited or reordered input cannot cause a silent skip.

The invariant is demonstrated rather than asserted: the tests interrupt a run
at every point one can be interrupted and compare the finished store, byte for
byte, with a run that was never interrupted.

## Memory

One batch of source text and one batch of vectors are held at a time, and the
store is never loaded whole, so the bound holds for a store of any size. The
default batch size is **64 records**: at the format's 1 MiB source-text ceiling
that is 64 MiB of text, and 64 vectors at the 8192-dimension ceiling is under
5 MiB as Python floats. Operators with more memory can raise it; the point of
the default is that nobody has to.

## What this does not do yet

- **No network.** Nothing here opens a socket or loads weights, and a test
  asserts it. Remote providers arrive in 3b-5 behind an explicit opt-in. The
  `egress` block is written now, always, so a later remote run has nowhere to
  hide: a reader cannot mistake a missing block for a local run.
- **No quality claim.** A record written today has no `quality` block at all.
  `not_measured` is a claim about a measurement that was attempted and found
  insufficient; nothing was attempted, so the honest output is silence. The
  comparison arrives in 3b-4.
- **No cost estimate.** 3b-3.

Records whose source text is `missing` or `deleted_by_policy` have no text to
send to any model. They are never embedded, never appear in the rebuilt store,
and are counted in the record as deliberately skipped.
