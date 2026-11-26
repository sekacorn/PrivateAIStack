# CLI

The CLI command is `privateaistack`.

Exit codes:

- `0`: success.
- `1`: runtime failure.
- `2`: invalid input or invalid configuration.
- `3`: required dependency unavailable.
- `4`: partial or degraded state.

Commands:

```bash
privateaistack --version
privateaistack doctor --directory .
privateaistack init ./deployment
privateaistack config check --directory .
privateaistack up --directory .
privateaistack up --directory . --observability
privateaistack down --directory .
privateaistack status --directory .
privateaistack logs --directory .
privateaistack health --directory .
privateaistack model list --directory .
privateaistack model pull --directory . --model qwen2.5:3b
privateaistack task run --directory . "Summarize the operating plan."
privateaistack knowledge add --directory . --content "Local document text"
privateaistack knowledge search --directory . "local document"
privateaistack memory export ./exports/memory
privateaistack review --directory . /app/sample-target
privateaistack audit show --directory .
privateaistack audit export --directory .
privateaistack audit verify --directory .
privateaistack backup --directory .
privateaistack restore --directory . backups/private_ai_stack.sql
```

The model commands operate inside the Docker Compose Ollama service by default. A model installed in a host Ollama service is not automatically present in the Compose named volume.

Commands use subprocess argument arrays and do not interpolate shell strings.

## `memory export`

Writes the knowledge store as `agent-memory.v1`, the portable format the AI
Infrastructure Suite reads and scores for the `memory` lock-in vector.

```bash
docker compose exec api privateaistack memory export /app/exports/memory-2026-11-25
```

It reads the store configured for the process it runs in (`DATABASE_URL` and the
other settings, exactly as the API reads them). Run it where that database is
reachable, which in the Compose deployment means inside the `api` container.
The target must be a new or empty directory that is not a link.

What it writes:

```text
agent-memory.json                       the document, with digests of everything below
items/0000.jsonl ...                    one chunk per line
embeddings/embedding-portability.json   the vectors, scheme privateaistack.hashed-bow.sha256-signed
embeddings/vectors/0000.jsonl ...
embeddings/sources/<sha256>.txt         each chunk's exact text
```

Each chunk is one item, and its `item_id` is the chunk ID. Its metadata holds
`document_id`, `source_name` and `document_sha256`, with the document's own
metadata under `metadata`. Vectors are bound to items by the SHA-256 of the
exported text.

The export is refused, with nothing written, when:

- **A chunk looks like it holds a credential.** The error names the chunk and
  the kind of credential, never the value. If it is a real credential, remove
  the document from the store first. If it is a false positive, there are two
  ways to proceed, and both are recorded:
  - `--redact-secrets`: every detected credential becomes
    `[REDACTED:<kind>]`. The document records `redaction_profile: "secrets"`,
    and the vectors are computed over the redacted text.
  - `--exclude CHUNK --exclude-reason "..."` (`--exclude` is repeatable): the
    chunk is left out of this export but stays in the store. Each exclusion is
    written to the audit log as `memory.export_exclusion` together with the
    export's digest.

  There is no option that exports a detected credential.
- **A stored vector is not the store's embedding of its chunk**, for example
  after a change of `EMBEDDING_DIMENSIONS`. Re-ingest those documents, or pass
  `--no-embeddings` to export the text alone.
- **The data is outside the format's bounds**: a chunk over 48 KiB, metadata
  nested deeper than 8 levels, and similar limits.

Other options: `--store-id`, `--export-id` and `--actor` (who is exporting,
recorded in the audit log).

The store keeps no deletion markers, so an export never carries tombstones. It
cannot show that a document was deleted.

`MemoryStore.export_jsonl` still exists but is superseded. Its output has no
version, no schema and no digests.

Exit codes: `0` success, `2` refused (invalid input), `1` runtime failure such
as an unreachable database.

## `reembed`

Rebuilding a vector store with a different embedding model, and comparing what
the rebuild retrieves. These commands do not touch a deployment: they read
`embedding-portability.v1` evidence documents and write files. The default
embedder is registry-owned deterministic package code; this is not a claim of
OS-level network isolation. Any injected executor requires a validated
`external_unverified` preflight before source files are read.

```bash
privateaistack reembed run --help
privateaistack reembed quality --help
```

Both exit `0` on success and `2` on invalid input, which includes an unknown
store identifier, a model identity this package does not implement, a document
that is not the schema it claims, and a file whose bytes no longer match a
digest the document declares for them.

### `reembed run`

Rebuilds one store out of an evidence document with a different model, in
resumable batches, and writes the rebuilt store, its own
`embedding-portability.v1` document, a `re-embed-record.v1` and a checkpoint.

```bash
privateaistack reembed run ./evidence/embedding-portability.json   --source-store support-memory   --target-directory ./rebuilt   --target-store support-memory-rebuilt   --model-id privateaistack.hashed-bow.sha256-signed   --model-revision 1   --dimensions 64   --availability open_weight_self_hostable
```

| Option | Means |
|---|---|
| `--source-store` | Which store in the document to rebuild. |
| `--target-directory` | Where to write the rebuilt store. |
| `--target-store` | Identifier for the rebuilt store. |
| `--model-id` | Target model id. Built in: `agentforge.hashed-bow.blake2b`, `privateaistack.hashed-bow.sha256-signed`. |
| `--model-revision` | Target model revision. |
| `--dimensions` | Target model dimensions. **Never defaulted**, because model identity is the triple id + revision + dimensions and the same scheme at two dimensions is two models. |
| `--availability` | `open_weight_self_hostable` or `vendor_only`. An operator attestation no offline check can confirm, so it has no default either. |
| `--batch-size` | Records per batch and per shard. Default 64. |

The egress options below decide whether anything may leave this machine. Every
default is the safe one, so a command that omits all of them cannot send
anything:

| Option | Default | Means |
|---|---|---|
| `--hosting-mode` | `local` | Declared executor location, `local` or `hosted`. **Not an isolation claim**: `local` means no socket is opened by package code, not that injected code cannot open one. |
| `--provider` | none | External executor identifier. Required for, and only for, an external executor; naming one on a registry-owned run is refused. |
| `--endpoint` | none | Declared endpoint. Normalised to an origin and validated, but an injected executor can ignore it, so actual routing stays attested. |
| `--allow-external` | **off** | Explicit consent to invoke an injected executor. Without it the run is refused before the evidence document is opened. |
| `--allow-hosted` | **off** | Additional consent for hosted execution. Both consents are needed for a hosted run; neither implies the other. |
| `--data-classification` | `unknown` | Operator-attested classification. Hosted `unknown`, `confidential` and `restricted` data is blocked, so the default is the blocking one. |
| `--telemetry-enabled` | off | Records external provider telemetry as an attested warning. Warned about even on a local run. |

There is no `--redaction-applied`: no redaction transform exists, so nothing
may claim one. There are no proxy or redirect options for the same reason —
a declaration this package cannot enforce would read as a control.

This package ships no hosted provider, so an allowed external run needs an
embedder supplied in code and exits `2` from the CLI with a message saying so.
[re-embedding.md](re-embedding.md) has the full `egress` block, the trust
boundaries, and why a submission count is never a delivery count.

Re-running the same command against the same target directory resumes an
interrupted run rather than starting over, and reports `resumed: true`. A batch
whose shard reached disk but was never claimed by the checkpoint is discarded
and re-embedded, never trusted.

### `reembed run --estimate-only`

Embeds a small calibration sample, measures it, extrapolates to the whole
store, and exits **without writing any shard, checkpoint or rebuilt store**.

```bash
privateaistack reembed run ./evidence/embedding-portability.json   --source-store support-memory   --target-directory ./rebuilt --target-store support-memory-rebuilt   --model-id privateaistack.hashed-bow.sha256-signed   --model-revision 1 --dimensions 64   --availability open_weight_self_hostable   --estimate-only --sample-size 16 --cost-profile ./cost-profile.json
```

| Option | Means |
|---|---|
| `--estimate-only` | Report the estimate and exit. Nothing is rebuilt. |
| `--sample-size` | Calibration sample size. Default 16. `0` reports no estimate rather than a guess. |
| `--sample-seed` | Seed for the sample selection, recorded in the estimate. |
| `--cost-profile` | A local cost profile as JSON, in AIMeter's shape. **Without one, no money figure is reported at all.** |
| `--estimate-output` | Also write the `re-embed-estimate.v1` document here. |

Only `records_to_embed` is exact. Every other figure is extrapolated from the
sample and is not a quotation. Measured latency is not portable to another
machine, so the estimate is deliberately not reproducible and is never read as
evidence by `exit-report`.

Money arithmetic is delegated to AIMeter and never reimplemented here. Install
it with `pip install "privateaistack[cost]"`; without it the estimate says so
rather than guessing. See [re-embedding.md](re-embedding.md) for the
`total_cost_precision` values, including why a rounded-to-zero cost is reported
as below reporting precision with a bound rather than as free.

### `reembed quality`

Ranks both stores for every query and reports what agreed, optionally writing
the result into the run's `re-embed-record.v1`.

```bash
privateaistack reembed quality ./evidence/embedding-portability.json ./rebuilt/embedding-portability.json   --before-store support-memory --after-store support-memory-rebuilt   --top-k 10 --min-top-k-overlap 0.7   --record ./rebuilt/re-embed-record.json
```

| Option | Means |
|---|---|
| `--before-store`, `--after-store` | Which store in each document. |
| `--query-set` | A `golden-query-set.v1` document. A file is always operator-chosen, so the result reports in full and **can never move a lock-in score**. |
| `--anchors` | How many anchors to draw by seed when no query set is supplied. Default 32. |
| `--anchor-seed` | **Override** the derived seed. The seed is otherwise `sha256` over the two store digests, so nobody can grind it. Passing one is a choice, so the query set reports as `operator` and moves no lock-in score. |
| `--top-k` | How many neighbours each ranking keeps. Default 10. |
| `--candidate-pool` | How many records each query is ranked against. Default 256. A `top_k_overlap` is a share of this pool, not of the whole store. |
| `--min-sample-size` | Below this many evaluated queries the result is `not_measured`. Default 20. |
| `--min-top-k-overlap` | Agreement floor. **No default**: a floor this package invented would be a bar nobody set. |
| `--min-rank-correlation` | Agreement floor. No default, same reason. |
| `--record` | A `re-embed-record.v1` to write the quality block into, replacing any earlier one. |

The default mode is record-anchored, which measures **neighborhood agreement
and never search quality**. Declare at least one agreement floor to get a
verdict. [re-embedding.md](re-embedding.md) has the statuses, the metrics, how
ties and empty query sets are handled, and why only a seeded draw can move a
score.

Reading a record this command writes needs ai-infrastructure-suite 0.1.0b3 or
later.
