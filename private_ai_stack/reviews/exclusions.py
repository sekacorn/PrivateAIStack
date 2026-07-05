from pathlib import Path

DEFAULT_EXCLUDED_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".env",
    ".ssh",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "htmlcov",
    ".coverage",
    "coverage",
}

SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tar", ".exe", ".dll", ".so", ".dylib", ".pyc"}


def is_excluded(path: Path, root: Path, max_bytes: int) -> tuple[bool, str | None]:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True, "outside_root"
    lowered_parts = {part.lower() for part in rel.parts}
    if lowered_parts & DEFAULT_EXCLUDED_PARTS:
        return True, "excluded_path"
    if path.suffix.lower() in SECRET_SUFFIXES:
        return True, "secret_suffix"
    if path.suffix.lower() in BINARY_SUFFIXES:
        return True, "binary_suffix"
    try:
        if path.stat().st_size > max_bytes:
            return True, "file_too_large"
    except OSError:
        return True, "unreadable"
    return False, None
