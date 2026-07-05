import json
from pathlib import Path
from typing import Any

from private_ai_stack.reviews.findings import Finding


def markdown_report(summary: dict[str, Any], findings: list[Finding]) -> str:
    """Render the human-readable review report without adding unverified claims."""
    lines = [
        "# PrivateAIStack Code Review Report",
        "",
        "## Technical Quality and Risk",
        "",
        f"- Files reviewed: {summary['files_reviewed']}",
        f"- Files excluded: {summary['files_excluded']}",
        f"- Findings: {summary['finding_count']}",
        "",
        "## Findings",
        "",
    ]
    if not findings:
        lines.append("No deterministic findings were produced. This does not prove the repository is secure or correct.")
    for finding in findings:
        location = f"{finding.file}:{finding.start_line}" if finding.file and finding.start_line else finding.file or "repository"
        lines.extend(
            [
                f"### {finding.severity.upper()} - {finding.title}",
                "",
                f"- Source: `{finding.source_tool}`",
                f"- Confidence: `{finding.confidence}`",
                f"- Location: `{location}`",
                f"- Status: `{finding.status}`",
                "",
                finding.explanation,
                "",
                f"Remediation: {finding.remediation}",
                "",
            ]
        )
    lines.extend(["## Tools That Did Not Run", ""])
    for tool in summary["tools"]:
        if tool["status"] == "not_run":
            lines.append(f"- `{tool['tool']}`: {tool.get('reason')}")
    return "\n".join(lines) + "\n"


def json_report(summary: dict[str, Any], findings: list[Finding]) -> dict[str, Any]:
    return {"summary": summary, "findings": [finding.model_dump() for finding in findings]}


def sarif_report(findings: list[Finding]) -> dict[str, Any]:
    """Emit a minimal SARIF document for file-specific findings."""
    return {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {"driver": {"name": "PrivateAIStack", "informationUri": "https://github.com/sekacorn/PrivateAIStack"}},
                "results": [
                    {
                        "ruleId": finding.id,
                        "level": "error"
                        if finding.severity in {"critical", "high"}
                        else "warning"
                        if finding.severity == "medium"
                        else "note",
                        "message": {"text": finding.title},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": finding.file or "repository"},
                                    "region": {"startLine": finding.start_line or 1},
                                }
                            }
                        ],
                    }
                    for finding in findings
                    if finding.file
                ],
            }
        ],
    }


def write_reports(output_dir: Path, review_id: str, summary: dict[str, Any], findings: list[Finding]) -> dict[str, Path]:
    review_dir = output_dir / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    markdown = review_dir / "report.md"
    json_path = review_dir / "report.json"
    sarif = review_dir / "report.sarif"
    markdown.write_text(markdown_report(summary, findings), encoding="utf-8")
    json_path.write_text(json.dumps(json_report(summary, findings), indent=2, default=str), encoding="utf-8")
    sarif.write_text(json.dumps(sarif_report(findings), indent=2), encoding="utf-8")
    return {"markdown": markdown, "json": json_path, "sarif": sarif}
