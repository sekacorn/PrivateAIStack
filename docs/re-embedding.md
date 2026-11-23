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

## Estimating a run before committing to it

```powershell
privateaistack reembed run evidence/embedding-portability.json --source-store support-memory --target-directory rebuilt --target-store support-memory-rebuilt --model-id privateaistack.hashed-bow.sha256-signed --model-revision 1 --dimensions 64 --availability open_weight_self_hostable --estimate-only
```

`--estimate-only` embeds a small **calibration sample**, measures it, and
extrapolates to the whole store as a `re-embed-estimate.v1` document. No shard,
no checkpoint and no rebuilt store are written, and no vector is kept.

The sample exists because AIMeter cannot forecast.
`calculate_local_inference_cost` needs an *observed* `performance.latency_ms`
greater than zero and `calculate_provider_cost` is token-driven, so neither can
be called before a run. Measuring a few records is the only honest way to get
a number, and the estimate reports which part is which:

| Field | Exactness |
|---|---|
| `records_to_embed` | exact, from the evidence document |
| `calibration` | measured, on this machine, for `sample_size` records |
| `projection` | extrapolated from the per-record means |
| `cost` | AIMeter's, when a profile was supplied; otherwise a stated reason there is none |
| `limits` | what the estimate does not claim, always including that it is not a quotation |

`--sample-size 0` reports **no estimate at all** rather than a fabricated one.
A sample larger than the store is capped at the store.

Records are selected by `sha256(seed + ':' + record_id)` order rather than by
identifier, because nothing says identifiers are unrelated to record size, and
taking the first N would be reproducibly biased. The seed is recorded so the
selection can be repeated.

Tokens are counted by asking the embedder for the tokens it will actually hash,
so a count cannot drift from the vector it describes. An embedder that does not
expose its tokenisation gets a whitespace count that says so on its face.

### Cost in money comes from AIMeter, or not at all

```powershell
python -m pip install "privateaistack[cost]"
```

Pass `--cost-profile profile.json`, in the shape AIMeter's
`local_cost_profile_catalog()` emits, and the money arithmetic is delegated to
`aimeter-oss` and credited in the document. AIMeter is optional, behind the
`[cost]` extra above, whose range matches the AI Infrastructure Suite's so a
machine with both resolves one version of AIMeter rather than two. Without it
installed the estimate says exactly that and reports records, tokens and
latency instead. Nothing here reimplements that arithmetic, and a profile's own
assumptions are the operator's: AIMeter's templates state on their face that
they are examples to be replaced.

**A cost that prints as zero is not a cost of zero.** Embedding a handful of
short records with a hashing scheme takes microseconds, so the total rounds to
`0.000000`, which a reader could take for free. The estimate distinguishes the
two cases:

| `total_cost_precision` | Means |
|---|---|
| `below_reporting_precision` | Above zero but smaller than the smallest unit AIMeter reports. `total_cost_bound` gives that bound, for example `< 0.000001 USD`, and the limits list says it is not free. |
| `exact_zero` | The profile declares no positive rate at all, so the cost really is nothing. |

Neither appears when the total is positive. The bound is derived from the
number of decimals AIMeter itself printed, so it stays correct if that
precision ever changes. A store with no records to embed reports
`not_estimated` as before: nothing to embed is nothing to extrapolate, which is
a different statement from a cost too small to print.

The estimate is a **forecast, not evidence**. `exit-report` never reads it, it
carries no digest of itself, and it is deliberately not reproducible: measured
latency depends on the machine and its load at the time. Only the counts in it
are stable.

## Comparing retrieval before and after

Rebuilding answers "can this be rebuilt?". `reembed quality` answers the
question that follows: do the two stores still put the same records near each
other?

```
privateaistack reembed quality before/embedding-portability.json after/embedding-portability.json   --before-store old-store --after-store new-store   --top-k 10 --min-top-k-overlap 0.7   --record after/re-embed-record.json
```

### It measures neighborhood agreement, not search quality

The default mode is `record_anchored`: a drawn record's own stored vector is
the query in both stores. That is not a shortcut. Ranking the *old* store for a
free-text query means embedding that query with the *old* model, which at a
lock-in score of 50 is exactly the model you no longer have. An anchor needs no
model at query time, so the comparison works in the case the whole phase exists
for.

The price is what the numbers mean. They say the two stores place the same
records near each other. They say **nothing** about whether a person searching
either store finds what they wanted. The record carries
`metric_basis: neighborhood_agreement` so no reader has to infer it, and the AI
Infrastructure Suite refuses a record-anchored result that claims otherwise.

An anchor is excluded from its own candidate list. It is its own nearest
neighbour in both stores, so leaving it in would add one guaranteed agreement
to every single query and quietly lift every overlap.

`text` mode embeds free-text queries with each store's own model. It needs both
model identities to be built-in schemes this package implements, because that
is the only case where the old model can be re-run offline. With
`expected_records` it measures relevance and reports
`metric_basis: retrieval_relevance`; without them it is agreement again.

### The metrics

| Metric | Means |
|---|---|
| `top_k_overlap` | Mean share of the old top-k that is still in the new top-k. A share of the candidate pool, not of the whole store. |
| `rank_correlation` | Mean Spearman correlation over the candidate pool, with tied similarities taking averaged ranks. |
| `hit_rate`, `hit_rate_before` | Share of queries whose expected records appear in the top-k. Text mode with expected records only. |
| `mrr`, `mrr_before` | Mean reciprocal rank of the first expected record. Text mode with expected records only. |

### Who chose the queries decides whether a score may move

| `anchor_selection` | Where the set came from | Reports | Can move 50 → 25 |
|---|---|---|---|
| `seeded_random` | Drawn by this tool from `--anchor-seed`, which is recorded | In full | Yes, when every other condition holds |
| `operator` | A `golden-query-set.v1` file you supplied | In full | Never |

A file is always `operator`, whatever it says about itself: the file is what
you chose to hand over. An operator who may pick the queries may pick the
result, and no offline check can tell a fair hand-picked set from a flattering
one, so a hand-picked set is reported with its full label, metrics and checks
and moves nothing. The suite reads the same rule from the record rather than
trusting this tool to have applied it.

### You declare the thresholds

There are no built-in agreement floors. A floor this package invented would be
a bar nobody set, and a verdict against it would be this tool's opinion in your
voice.

A run with no floor still gets its numbers. It reports `status: unjudged`: the
metrics are real, the verdict does not exist, and the notes name what to
declare to get one — `--min-top-k-overlap`, `--min-rank-correlation`, or
`expected_records` in the query set so the hit-rate and MRR drop caps apply.
`unjudged` can never move a lock-in score, whoever drew the queries, because
there is no verdict for a score to move on.

| Status | Means | Can move 50 → 25 |
|---|---|---|
| `sampled` | Every declared threshold was met. | Yes, with the other four conditions |
| `below_threshold` | At least one declared threshold was not met. | No |
| `unjudged` | Metrics measured, no agreement floor declared, so no verdict. | No |
| `not_measured` | No query set, none usable, or fewer queries than `--min-sample-size`. | No |

The bar travels with the verdict. Every declared threshold is written into the
record's `thresholds` and listed among the suite's `attested_claims`, so no
reader sees `sampled` without seeing what was cleared. `minimum_sample_size`
defaults to 20, the same number ModelSwapBench uses, so a calibration done for
one is not wasted on the other.

### Ties, empty sets and other awkward cases

| Situation | What happens |
|---|---|
| Equal similarities | Ordered by ascending `record_id`, recorded in `tie_break`, so a reader can tell a tie from a disagreement. |
| The top-k boundary falls inside a tie | Counted in `boundary_ties`. Membership there was decided by the tie break, not by the vectors. |
| Every candidate tied for a query | Rank correlation is undefined for it, `correlation_undefined` is set, and no 0 is invented. A 0 would read as disagreement rather than as absence. |
| No agreement floor declared | `unjudged`: the metrics are reported and the notes name what to declare. |
| No query could be evaluated | `not_measured`, no metrics, and a note saying it is not a passing result. |
| Fewer queries than `--min-sample-size` | `not_measured` with `sufficient_evidence: false`. The comparison ran; it is not reported as a measurement. |
| An anchor absent from either store | Counted in `queries_skipped`, not fatal. |
| `--top-k` larger than the pool | The effective k is smaller and `top_k_capped` is set. |
| The two stores share no record identifiers | Refused. Comparing neighborhoods across different populations gives a meaningless number, not a low one. |

### Bounds

Every query is ranked against a **candidate pool** rather than the whole store,
drawn by the same seed and defaulting to 256 records. Every anchor is kept in
the pool whatever the draw says, so an anchor is never missing from the pool it
is ranked inside. The pool size is written into the record, because an overlap
over 256 drawn records is a different claim from an overlap over the whole
store.

Vectors are held as packed doubles, capped at eight million values per store
(64 MiB, 128 MiB for both), and a comparison that would need more than 200
million multiplications is refused with a message rather than attempted. All of
this is so a run stays inside a 16 GB machine beside everything else.

### Reading the result needs a current suite

The fields beyond `status`, `query_set_size` and `top_k` are additive within
`re-embed-record.v1`, so the version does not change, but the suite's quality
block refuses unknown fields. Reading a record this command writes needs
**ai-infrastructure-suite 0.1.0b3 or later**.

## What leaves the machine

Re-embedding sends source text to a model, which the evidence work in 3a never
did. That is the sharpest new risk in the phase, so it has a gate rather than a
convention.

**The default is local.** Nothing opens a socket, and tests assert that by
replacing `socket.socket` with something that fails the test if anything
touches it, rather than by trusting that nothing did.

**A hosted run needs two things said out loud**, and a third that this package
does not provide:

```
privateaistack reembed run ./evidence/embedding-portability.json ...   --hosting-mode hosted --provider example-hosted   --data-classification confidential --allow-hosted
```

| Flag | Default | Means |
|---|---|---|
| `--hosting-mode` | `local` | `local` opens no socket at all. |
| `--provider` | none | Required for, and only for, a hosted run. Naming one on a local run is refused. |
| `--allow-hosted` | off | The explicit opt-in. Without it a hosted run is refused **before the evidence document is opened**. |
| `--data-classification` | `unknown` | The operator's classification. Attested, never checked. |
| `--redaction-applied` | off | An attestation that the text was redacted. Attested, never checked. |
| `--telemetry-enabled` | off | Recorded, and warned about even on a local run. |

The refusal happens before any file is read, so it cannot depend on anything
the input says — a test passes a path that does not exist and still gets the
egress refusal, for the right reason.

**This package ships no hosted provider.** An allowed hosted run needs an
embedder supplied in code. Falling back to a local model would make the record
say `hosted` about a run that was not, so it is refused instead. From the CLI a
hosted run therefore always stops at that boundary, which is the honest place
for an operator to meet it.

### Every run records what left

The `egress` block is written on every run, so its absence can never be
mistaken for a local run. A hosted run records the provider, the count, whether
raw text went, the opt-in, the classification, the redaction claim, the
telemetry state, and the preflight warnings the operator was shown.

`records_transmitted` counts records actually embedded, not planned: records
whose source text is `missing` or `deleted_by_policy` have nothing to send, so
counting them would over-report what left.

Two of those fields are attested and appear in the suite's `attested_claims`.
No scanner can read source text and tell whether an organisation considers it
confidential, or whether a redaction removed what the operator thinks it
removed.

The rule that a run cannot report transmitting content nobody enabled lives in
the suite's reader as well as in this gate, so a tool that skipped its own
check still cannot produce a readable record of having done so.

### The warnings are the same sentences ModelSwapBench uses

`hosted execution is blocked until explicitly enabled`,
`private unredacted dataset content would be transmitted to a hosted provider`
and `telemetry is enabled; review exported fields before execution` are
restated here verbatim, because the two projects take no dependency on each
other and an operator who has read one tool's warning should not have to learn
a second wording for the same risk. A test pins the strings.

One wording is worth naming rather than quietly changing: the unredacted
warning is silenced by a classification of `public` **or** `unknown`, which is
what ModelSwapBench does. `unknown` is the weaker of the two, and it is also
the default, so a hosted run with no classification given does not raise that
warning. Changing it would mean the two tools disagreeing, which is the worse
problem.

## What this does not do yet

- **No hosted provider ships here.** The gate, the seam and the record are
  here; choosing a vendor, writing its client and taking a runtime dependency
  on an HTTP library are decisions for whoever is doing the sending. A hosted
  run needs an embedder supplied in code and is refused without one.
- **No quality claim from `reembed run`.** A record `run` writes has no
  `quality` block at all. `not_measured` is a claim about a measurement that
  was attempted and found insufficient; `run` attempts nothing, so the honest
  output is silence. `reembed quality` adds the block when it is asked to.
- **No remote cost estimate.** The local estimate landed in 3b-3; a remote
  provider's token cost needs the egress opt-in of 3b-5, because the
  calibration sample itself would transmit source text.

Records whose source text is `missing` or `deleted_by_policy` have no text to
send to any model. They are never embedded, never appear in the rebuilt store,
and are counted in the record as deliberately skipped.
