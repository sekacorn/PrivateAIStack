from private_ai_stack.reviews.findings import Finding


class CodeQualityAgent:
    name = "CodeQualityAgent"

    def interpret(self, findings: list[Finding]) -> list[str]:
        return [
            f"{finding.file or 'repository'}: {finding.title}" for finding in findings if finding.category in {"code_quality", "tooling"}
        ][:10]
