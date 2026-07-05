from private_ai_stack.reviews.findings import ToolRun


class InfrastructureReviewAgent:
    name = "InfrastructureReviewAgent"

    def interpret(self, runs: list[ToolRun]) -> list[str]:
        interesting = [run for run in runs if run.tool in {"hadolint", "shellcheck", "yamllint"}]
        return [f"{run.tool}: {run.status}" for run in interesting]
