from pathlib import Path

from pydantic import BaseModel

from private_ai_stack.reviews.exclusions import is_excluded


class CollectedFile(BaseModel):
    path: Path
    relative_path: str
    size: int


class RepositorySnapshot(BaseModel):
    root: Path
    files: list[CollectedFile]
    excluded: dict[str, str]
    languages: list[str]


def collect_repository(repository_path: str, max_file_bytes: int) -> RepositorySnapshot:
    root = Path(repository_path).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Repository path does not exist or is not a directory: {repository_path}")

    files: list[CollectedFile] = []
    excluded: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        skip, reason = is_excluded(path, root, max_file_bytes)
        rel = path.relative_to(root).as_posix()
        if skip:
            excluded[rel] = reason or "excluded"
            continue
        files.append(CollectedFile(path=path, relative_path=rel, size=path.stat().st_size))

    suffixes = {file.path.suffix.lower() for file in files}
    languages: list[str] = []
    if ".py" in suffixes:
        languages.append("python")
    if {".yml", ".yaml"} & suffixes:
        languages.append("yaml")
    if ".md" in suffixes:
        languages.append("markdown")
    if ".sh" in suffixes:
        languages.append("shell")
    if any(file.path.name.lower() == "dockerfile" for file in files):
        languages.append("docker")
    return RepositorySnapshot(root=root, files=files, excluded=excluded, languages=languages or ["unknown"])
