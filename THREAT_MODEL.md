# PrivateAIStack Threat Model

Status: `0.2.0b1` Beta candidate threat model.

PrivateAIStack provides application-level governance for local AI workflows. It is not VM-grade isolation, a container escape prevention system, or a substitute for professional security review.

## Assets

- Local source repositories reviewed by the system.
- Local documents ingested into RAG.
- PostgreSQL memory and audit records.
- Model prompts and responses.
- Configuration and environment variables.
- Exported reports, memory, and audit JSONL.
- User approval decisions.
- OpenTelemetry traces.

## Primary Trust Boundaries

- User or CI to FastAPI.
- FastAPI container to reviewed repository read-only mount.
- FastAPI container to PostgreSQL internal network.
- FastAPI container to Ollama.
- FastAPI and Forge to OpenTelemetry Collector.
- OpenTelemetry Collector to Jaeger.
- Optional hosted provider boundary when explicitly enabled.

## Threats Covered

- Malicious reviewed repository.
- Malicious prompt.
- Prompt injection inside source code or documentation.
- Poisoned RAG documents.
- Secret exfiltration.
- Unrestricted tool invocation.
- Shell injection.
- SQL injection.
- SSRF.
- Excessive model or compute use.
- Audit-log tampering.
- Dependency compromise.
- Container escape.
- Exposed Ollama endpoint.
- Exposed PostgreSQL endpoint.
- Telemetry leakage.
- Malicious GitHub Actions changes.
- Unsafe approval defaults.

## Default Security Posture

- Safe-static review mode is default.
- Reviewed repositories are read-only by default.
- No Docker socket is mounted.
- No host root is mounted.
- No access to the user's home directory is required.
- `.env`, private keys, cloud credentials, `.git`, virtual environments, dependency folders, caches, build outputs, binaries, and large files are excluded by default.
- External network access requires explicit approval.
- Dependency installation requires explicit approval.
- Test execution in sandboxed mode requires explicit approval.
- Missing approvers deny by default.

## Review Modes

### safe-static

- Default.
- Read-only.
- No project dependency installation.
- No arbitrary test execution.
- No external network by default.
- Static tools and AI interpretation only.

### sandboxed-execution

- Intentionally unavailable in the current Beta candidate.
- The API returns an approval-required response rather than pretending to provide a sandbox.
- A future implementation must define explicit approval, isolation, resource, and network controls before this mode can be enabled.

## Telemetry Protections

Telemetry must not include:

- Raw secrets.
- Full sensitive source files.
- Unredacted prompts containing private code.
- Database passwords.
- API keys.
- Private keys or credential files.

Trace attributes should use IDs, counts, hashes, bounded filenames, and summarized status values.

## Known Limitations

- Local models can hallucinate.
- Static analysis has false positives and false negatives.
- Agent findings require human review.
- Application-level policy is not a complete sandbox.
- Running untrusted code is riskier than statically reviewing it.
- Safe-static mode cannot prove runtime behavior.
