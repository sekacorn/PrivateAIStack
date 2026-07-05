import json

from private_ai_stack.reviews.findings import Finding, ToolRun, make_fingerprint


def normalize_tool_runs(runs: list[ToolRun]) -> list[Finding]:
    findings: list[Finding] = []
    for run in runs:
        if run.status == "not_run":
            findings.append(
                _finding(
                    run.tool,
                    "tooling",
                    "informational",
                    "confirmed",
                    f"{run.tool} did not run",
                    run.reason or "Tool unavailable.",
                    None,
                    None,
                    run.reason or "",
                )
            )
            continue
        if run.tool == "ruff" and run.stdout.strip():
            findings.extend(_ruff(run))
        elif run.tool == "bandit" and run.stdout.strip():
            findings.extend(_bandit(run))
        elif run.status == "failed":
            findings.append(
                _finding(
                    run.tool,
                    "tooling",
                    "low",
                    "confirmed",
                    f"{run.tool} reported issues",
                    run.stderr or run.stdout or "Tool exited non-zero.",
                    None,
                    None,
                    run.stderr or run.stdout,
                )
            )
    return deduplicate(findings)


def deduplicate(findings: list[Finding]) -> list[Finding]:
    seen: set[str] = set()
    result: list[Finding] = []
    for finding in findings:
        if finding.fingerprint in seen:
            continue
        seen.add(finding.fingerprint)
        result.append(finding)
    return result


def _ruff(run: ToolRun) -> list[Finding]:
    try:
        payload = json.loads(run.stdout)
    except json.JSONDecodeError:
        return [
            _finding(
                "ruff", "code_quality", "low", "confirmed", "Ruff output could not be parsed", run.stdout[:1000], None, None, run.stdout
            )
        ]
    findings: list[Finding] = []
    for item in payload:
        file = item.get("filename")
        location = item.get("location") or {}
        line = location.get("row")
        title = f"Ruff {item.get('code', 'issue')}: {item.get('message', 'lint finding')}"
        findings.append(_finding("ruff", "code_quality", "low", "confirmed", title, item.get("message", ""), file, line, json.dumps(item)))
    return findings


def _bandit(run: ToolRun) -> list[Finding]:
    try:
        payload = json.loads(run.stdout)
    except json.JSONDecodeError:
        return [
            _finding(
                "bandit", "security", "medium", "confirmed", "Bandit output could not be parsed", run.stdout[:1000], None, None, run.stdout
            )
        ]
    findings: list[Finding] = []
    for item in payload.get("results", []):
        severity = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}.get(str(item.get("issue_severity", "")).upper(), "medium")
        confidence = {"LOW": "medium", "MEDIUM": "high", "HIGH": "confirmed"}.get(str(item.get("issue_confidence", "")).upper(), "medium")
        title = f"Bandit {item.get('test_id', 'issue')}: {item.get('issue_text', 'security finding')}"
        findings.append(
            _finding(
                "bandit",
                "security",
                severity,
                confidence,
                title,
                item.get("issue_text", ""),
                item.get("filename"),
                item.get("line_number"),
                json.dumps(item),
            )
        )
    return findings


def _finding(
    source_tool: str,
    category: str,
    severity: str,
    confidence: str,
    title: str,
    explanation: str,
    file: str | None,
    line: int | None,
    evidence: str,
) -> Finding:
    fingerprint = make_fingerprint(source_tool, title, file, line)
    return Finding(
        id=f"{source_tool}-{fingerprint}",
        source_tool=source_tool,
        agent="deterministic_tool",
        category=category,
        severity=severity,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        title=title,
        explanation=explanation,
        file=file,
        start_line=line,
        end_line=line,
        evidence=evidence[:4000],
        remediation="Review the cited evidence and address the underlying issue; rerun the deterministic tool afterward.",
        fingerprint=fingerprint,
    )
