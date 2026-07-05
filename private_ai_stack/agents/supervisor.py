from private_ai_stack.agents.code_quality import CodeQualityAgent
from private_ai_stack.agents.documentation import DocumentationReviewAgent
from private_ai_stack.agents.infrastructure import InfrastructureReviewAgent
from private_ai_stack.agents.security import SecurityReviewAgent
from private_ai_stack.agents.tests import TestReviewAgent
from private_ai_stack.reviews.collector import RepositorySnapshot
from private_ai_stack.reviews.findings import Finding, ToolRun


class ReviewSupervisor:
    name = "ReviewSupervisor"

    def summarize(self, snapshot: RepositorySnapshot, findings: list[Finding], runs: list[ToolRun]) -> dict[str, object]:
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4}
        ranked = sorted(findings, key=lambda item: severity_order[item.severity])
        return {
            "repository": str(snapshot.root),
            "languages": snapshot.languages,
            "files_reviewed": len(snapshot.files),
            "files_excluded": len(snapshot.excluded),
            "finding_count": len(findings),
            "top_findings": [finding.model_dump() for finding in ranked[:10]],
            "tools": [run.model_dump() for run in runs],
            "technical_quality_and_risk": {
                "code_quality": CodeQualityAgent().interpret(findings),
                "security": SecurityReviewAgent().interpret(findings),
                "tests": TestReviewAgent().interpret(runs),
                "infrastructure": InfrastructureReviewAgent().interpret(runs),
                "documentation": DocumentationReviewAgent().interpret(snapshot),
            },
        }
