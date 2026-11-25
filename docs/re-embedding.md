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
| `seeded_random` | Drawn from the **derived** seed, which is recorded | In full | Yes, when every other condition holds |
| `operator` | A `golden-query-set.v1` file, or a seed you chose with `--anchor-seed` | In full | Never |

#### The seed is derived, not chosen

An operator who picks the seed can grind it: draw query set after query set
until the numbers clear the bar, then publish the one that passed. So the seed
is `sha256` over the source and target store digests — content the record
already commits to, which nobody can steer and a third party can recompute.

Passing `--anchor-seed` overrides it, and that is treated as a choice: the set
reports as `operator`, moves no score, and no seed is written into the record.
The anchors drawn are identical either way; only the provenance differs, and
provenance is the whole question.

The AI Infrastructure Suite derives the same value independently and reads any
other seed as an operator-chosen set. A cross-repo test asserts both sides
compute it identically, because if they ever diverged every record this tool
writes would silently stop being able to move a score.

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

## Executor trust and data movement

Re-embedding passes source text to an embedder. The privacy gate distinguishes
who owns that code before it reads the evidence document or any source file:

- `local_builtin` is selected from this package's registry. The implementation
  is deterministic package code. This is not an OS-level network-isolation
  claim.
- `external_unverified` is any injected executor, including an object whose
  class resembles a built-in, a local service, a provider SDK, or hosted code.
  Its endpoint, proxy, redirect, telemetry, and network behaviour are operator
  attestations that this package cannot verify.

An injected executor needs `allow_external=True`. Hosted execution additionally
needs `allow_hosted=True`, an HTTPS endpoint declaration, and a `public` or
`internal` classification. `unknown`, `confidential`, and `restricted` source
data are blocked from hosted execution even when both consent flags are true.
The refusal occurs before the evidence document is opened.

### "Local" here means loopback only, on purpose

This gate is stricter than the rule the rest of the suite uses to classify an
endpoint. AgentForge, ModelSwapBench and the AI Infrastructure Suite treat a
private-network address (`10.x`, `192.168.x`, link-local) as **local**, because
their question is whether a call leaves your own infrastructure. This gate asks
a narrower question, whether **source text leaves this machine**, so:

- a `local` external endpoint must be `localhost` or a loopback address;
- a `hosted` external endpoint must use HTTPS and a public address;
- a private-network address is therefore accepted as neither. An embedder on
  another machine in your network is not on this machine, and this package
  will not describe sending source text to it as local. It is also not a public
  HTTPS endpoint, so it cannot be declared hosted.

That gap is deliberate, not an oversight. If you need to re-embed against an
embedder elsewhere on a private network, expose it on loopback (a tunnel or a
local proxy you control) so the declaration matches what the record says.

Two smaller differences follow from the same strictness. The recorded
`endpoint_origin` drops a default port (`https://host`, not `https://host:443`),
and a query string or fragment in the endpoint is refused, as credentials are.

The CLI exposes the same preflight flags, but it does not ship an injected or
hosted provider, so an allowed external CLI run stops before reading source
data and explains that an injected embedder is required.

| Flag | Default | Meaning |
|---|---|---|
| `--hosting-mode` | `local` | Declared location. It is not proof of isolation. |
| `--provider` | none | External executor identifier. |
| `--endpoint` | none | Declared endpoint; only its normalized origin is retained. Actual use remains attested. |
| `--allow-external` | off | Explicit permission to invoke injected code. |
| `--allow-hosted` | off | Additional permission for hosted execution. |
| `--data-classification` | `unknown` | Operator-attested classification. Hosted `unknown`, `confidential`, and `restricted` data is blocked. |
| `--telemetry-enabled` | off | External provider telemetry attestation and warning. |

No redaction transform exists in this checkpoint. A caller therefore cannot
claim source text was redacted. Proxy routing and redirect following are also
unsupported declarations. Arbitrary external code could still use them, which
is why its network behaviour stays `attested` rather than verified.

### Evidence semantics

Every run writes an `egress` block. `records_submitted_to_external_executor` is
derived from distinct records in completed batches. A hosted run copies that
actual count to `records_transmitted` and records
`raw_source_text_transmitted: true`; no operator flag can turn it false.
`missing` and `deleted_by_policy` records never enter a batch and are never
submitted.

#### Submission is not delivery

This package hands a payload to a callable. It cannot observe what that
callable does next, so it never knows that a provider received anything.

`records_transmitted` keeps its established name because readers and the
shipped schema already use it, but it is **submission accounting**, and the
qualifier travels with the number rather than living only in prose:

| Field | Value | Says |
|---|---|---|
| `transmission_basis` | `handed_to_executor` | The count is records this pipeline passed to the executor callable. |
| `transmission_basis` | `no_transmission` | A registry-owned embedder ran; nothing was handed to anything. |
| `transmission_accounting` | `distinct_records_in_completed_batches` | Exactly what the number counts. |
| `transmission_attempts_may_exceed_count` | `true` | Attempts can be higher than the count. See below. |
| `delivery_verification` | `unverified` | No provider delivery was established. There is no other value. |

The last three appear only for an external executor, because a registry-owned
run has no handover to qualify.

The count is exact for what it names and is **not** a count of attempts. Two
things can push attempts above it, and both are stated rather than implied:
an executor may retry internally where this package cannot see it, and a crash
before a checkpoint reaches disk discards a batch whose records were already
handed over, so a resume hands them again. A test drives exactly that case and
asserts the re-handover happens while the distinct count stays correct.

Nothing in a written record can say a provider received a record. The operator
is also warned before the run: `records handed to an external executor are not
confirmed delivered to any provider`.

#### The suite has to be able to read this

PrivateAIStack and the AI Infrastructure Suite take no code dependency on each
other. The contract between them is the `re-embed-record.v1` document, and
nothing used to check it: this package came to emit twenty-one egress fields
against a reader that knew ten, so every record it wrote was rejected outright
and no test noticed.

The `suite-contract` CI job closes that. It installs the suite from its `main`
branch and validates **every** record variant this package emits — registry
owned, external-local and hosted — through both the suite's reader and its
shipped JSON Schema, failing on any field the installed suite does not know.
Both halves are checked because they fail differently: the reader enforces
cross-field consistency a schema cannot express, and the schema enforces the
field inventory through `additionalProperties: false`.

The job also asserts that an injected unknown field *is* refused by both, so a
pass means the check still has teeth rather than that it has quietly stopped
looking.

`suite-contract` is a **required** status check on `master`, and it also runs
on a nightly schedule. The suite is a separate repository on its own cadence,
so a change there can break this contract without anything happening here; the
nightly run surfaces that within a day rather than whenever somebody next opens
a pull request.

Because it is required, a missing suite is a **failure** in that job, not a
skip: `PRIVATEAISTACK_REQUIRE_SUITE_CONTRACT=1` turns the import into a hard
error. A botched install would otherwise skip every test and report the
required check as passing, which is the one outcome a contract gate must never
produce.

The suite is not a declared dependency of this package — that would invert the
layering — so without that variable `tests/contract` skips when it is absent
and the offline matrix stays offline. Run it locally with:

```
pip install -e ".[dev]" jsonschema
pip install "git+https://github.com/sekacorn/AI-Infrastructure-Suite.git@main"
pytest --no-cov tests/contract
```

#### Resume binding

Checkpoints are bound to the executor kind, provider, endpoint origin,
classification, and consent context. A resume cannot switch those declarations
and then attribute earlier batches to a different executor. The binding is by
digest, so the checkpoint does not echo the endpoint it commits to.

For `local_builtin`, the record says `registry_owned`. For
`external_unverified`, execution and network behaviour say `attested`. Neither
label proves authenticity, provenance, or OS-enforced network isolation.

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
  calibration with an injected executor uses the same consent gate and records
  how many sample payloads were submitted.

Records whose source text is `missing` or `deleted_by_policy` have no text to
send to any model. They are never embedded, never appear in the rebuilt store,
and are counted in the record as deliberately skipped.
