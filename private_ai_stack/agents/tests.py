from private_ai_stack.reviews.findings import ToolRun


class TestReviewAgent:
    name = "TestReviewAgent"

    def interpret(self, runs: list[ToolRun]) -> list[str]:
        pytest_runs = [run for run in runs if run.tool == "pytest"]
        if not pytest_runs:
            return ["Tests are not executed in safe-static mode unless explicitly approved."]
        return [f"pytest status: {run.status}" for run in pytest_runs]
