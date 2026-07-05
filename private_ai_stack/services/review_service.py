import hashlib
import uuid
from pathlib import Path

from private_ai_stack.agents.supervisor import ReviewSupervisor
from private_ai_stack.api.errors import AppError
from private_ai_stack.api.schemas import ReviewRequest, ReviewResponse, Status, utc_now
from private_ai_stack.audit.writer import AuditWriter
from private_ai_stack.config.settings import Settings
from private_ai_stack.reviews.collector import collect_repository
from private_ai_stack.reviews.exclusions import is_excluded
from private_ai_stack.reviews.findings import Finding
from private_ai_stack.reviews.normalizers import normalize_tool_runs
from private_ai_stack.reviews.reports import json_report, markdown_report, sarif_report, write_reports
from private_ai_stack.reviews.runner import run_static_tools


class ReviewService:
    def __init__(self, settings: Settings, audit: AuditWriter) -> None:
        self.settings = settings
        self.audit = audit
        self._reviews: dict[str, ReviewResponse] = {}
        self._findings: dict[str, list[Finding]] = {}
        self._reports: dict[str, dict[str, Path]] = {}
        self._summaries: dict[str, dict[str, object]] = {}

    async def create_review(self, payload: ReviewRequest, request_id: str, trace_id: str | None) -> ReviewResponse:
        if payload.mode == "sandboxed-execution":
            raise AppError("approval_required", "sandboxed-execution is experimental and requires an explicit approval interface.", 403)
        review_id = str(uuid.uuid4())
        now = utc_now()
        review = ReviewResponse(
            review_id=review_id, status=Status.running, request_id=request_id, trace_id=trace_id, created_at=now, updated_at=now
        )
        self._reviews[review_id] = review
        self.audit.write(
            "review.created",
            entity_type="review",
            entity_id=review_id,
            request_id=request_id,
            trace_id=trace_id,
            details={"repository_path": payload.repository_path, "mode": payload.mode},
        )
        try:
            before_hash = self._tree_hash(payload.repository_path)
            snapshot = collect_repository(payload.repository_path, self.settings.max_review_file_bytes)
            runs = run_static_tools(snapshot.root, "python" in snapshot.languages)
            findings = normalize_tool_runs(runs)
            summary = ReviewSupervisor().summarize(snapshot, findings, runs)
            paths = write_reports(self.settings.reports_dir, review_id, summary, findings)
            after_hash = self._tree_hash(payload.repository_path)
            unchanged = before_hash == after_hash
            summary["source_unchanged"] = unchanged
            self._findings[review_id] = findings
            self._reports[review_id] = paths
            self._summaries[review_id] = summary
            review.status = Status.succeeded
            review.summary = {"finding_count": len(findings), "source_unchanged": unchanged, "report_dir": str(paths["markdown"].parent)}
            self.audit.write(
                "review.succeeded",
                entity_type="review",
                entity_id=review_id,
                request_id=request_id,
                trace_id=trace_id,
                details={"finding_count": len(findings), "source_unchanged": unchanged, "report_hash": self._file_hash(paths["json"])},
            )
        except Exception as exc:
            review.status = Status.failed
            review.error = exc.__class__.__name__
            self.audit.write(
                "review.failed",
                entity_type="review",
                entity_id=review_id,
                request_id=request_id,
                trace_id=trace_id,
                details={"error": str(exc)},
            )
        review.updated_at = utc_now()
        return review

    def get_review(self, review_id: str) -> ReviewResponse:
        try:
            return self._reviews[review_id]
        except KeyError as exc:
            raise AppError("not_found", "Review not found.", 404, {"review_id": review_id}) from exc

    def get_findings(self, review_id: str) -> list[Finding]:
        self.get_review(review_id)
        return self._findings.get(review_id, [])

    def get_report(self, review_id: str, format: str) -> dict[str, object]:
        self.get_review(review_id)
        summary = self._summaries.get(review_id, {})
        findings = self._findings.get(review_id, [])
        if format == "json":
            return json_report(summary, findings)
        if format == "sarif":
            return sarif_report(findings)
        return {"format": "markdown", "content": markdown_report(summary, findings)}

    def _tree_hash(self, path: str) -> str:
        root = Path(path).resolve()
        digest = hashlib.sha256()
        for item in sorted(root.rglob("*")):
            if item.is_file() and not is_excluded(item, root, self.settings.max_review_file_bytes)[0]:
                digest.update(str(item.relative_to(root)).encode("utf-8"))
                digest.update(hashlib.sha256(item.read_bytes()).digest())
        return digest.hexdigest()

    def _file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
