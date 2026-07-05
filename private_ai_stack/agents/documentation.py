from private_ai_stack.reviews.collector import RepositorySnapshot


class DocumentationReviewAgent:
    name = "DocumentationReviewAgent"

    def interpret(self, snapshot: RepositorySnapshot) -> list[str]:
        names = {file.relative_path.lower() for file in snapshot.files}
        if "readme.md" not in names:
            return ["No README.md was collected; documentation drift cannot be fully assessed."]
        return ["README.md was collected; compare documented commands against generated review output."]
