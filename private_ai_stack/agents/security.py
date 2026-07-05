from private_ai_stack.reviews.findings import Finding


class SecurityReviewAgent:
    name = "SecurityReviewAgent"

    def interpret(self, findings: list[Finding]) -> list[str]:
        return [f"{finding.severity}: {finding.title}" for finding in findings if finding.category == "security"][:10]
