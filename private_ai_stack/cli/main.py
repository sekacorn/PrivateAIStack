from __future__ import annotations

import json
import shutil
import socket
import subprocess  # nosec B404
import sys
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import as_file, files
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from private_ai_stack import __version__
from private_ai_stack.audit.writer import AuditWriter

SUCCESS = 0
RUNTIME_FAILURE = 1
INVALID_INPUT = 2
DEPENDENCY_UNAVAILABLE = 3
DEGRADED = 4

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "qwen2.5:3b"


class CliError(Exception):
    def __init__(self, message: str, exit_code: int = RUNTIME_FAILURE) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


app = typer.Typer(help="Manage a local PrivateAIStack deployment.")
config_app = typer.Typer(help="Validate deployment configuration.")
model_app = typer.Typer(help="Manage Ollama models through the Compose service.")
task_app = typer.Typer(help="Run agent tasks through the API.")
knowledge_app = typer.Typer(help="Add and search local knowledge through the API.")
audit_app = typer.Typer(help="Inspect and export local audit logs.")
app.add_typer(config_app, name="config")
app.add_typer(model_app, name="model")
app.add_typer(task_app, name="task")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(audit_app, name="audit")


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"privateaistack {__version__}")
        raise typer.Exit(SUCCESS)


@app.callback()
def callback(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the PrivateAIStack CLI version.", callback=version_callback, is_eager=True),
    ] = False,
) -> None:
    del version


def main() -> None:
    app()


def fail(message: str, exit_code: int = RUNTIME_FAILURE) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(exit_code)


def handle_error(exc: Exception, verbose: bool = False) -> None:
    if isinstance(exc, typer.Exit):
        raise exc
    if verbose:
        raise exc
    if isinstance(exc, CliError):
        fail(str(exc), exc.exit_code)
    fail(str(exc), RUNTIME_FAILURE)


def ensure_directory(directory: Path) -> Path:
    return directory.expanduser().resolve()


def compose_file(directory: Path) -> Path:
    path = ensure_directory(directory) / "compose.yaml"
    if not path.exists():
        raise CliError(f"compose.yaml was not found in {path.parent}", INVALID_INPUT)
    return path


def run_command(args: list[str], cwd: Path | None = None, timeout: int | None = 120, stdin: str | None = None) -> CommandResult:
    try:
        completed = subprocess.run(  # nosec B603
            args, cwd=cwd, input=stdin, text=True, capture_output=True, timeout=timeout, check=False
        )
    except FileNotFoundError as exc:
        raise CliError(f"Required command is unavailable: {args[0]}", DEPENDENCY_UNAVAILABLE) from exc
    except subprocess.TimeoutExpired as exc:
        raise CliError(f"Command timed out: {' '.join(args)}", RUNTIME_FAILURE) from exc
    return CommandResult(args=args, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def stream_command(args: list[str], cwd: Path | None = None) -> CommandResult:
    try:
        completed = subprocess.run(args, cwd=cwd, timeout=None, check=False)  # nosec B603
    except FileNotFoundError as exc:
        raise CliError(f"Required command is unavailable: {args[0]}", DEPENDENCY_UNAVAILABLE) from exc
    return CommandResult(args=args, returncode=completed.returncode, stdout="", stderr="")


def run_command_with_file_stdin(args: list[str], input_path: Path, cwd: Path | None = None, timeout: int | None = 120) -> CommandResult:
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            completed = subprocess.run(  # nosec B603
                args,
                cwd=cwd,
                stdin=handle,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
    except FileNotFoundError as exc:
        raise CliError(f"Required command is unavailable: {args[0]}", DEPENDENCY_UNAVAILABLE) from exc
    except subprocess.TimeoutExpired as exc:
        raise CliError(f"Command timed out: {' '.join(args)}", RUNTIME_FAILURE) from exc
    return CommandResult(args=args, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def compose_args(directory: Path, *args: str, observability: bool = False) -> list[str]:
    command = ["docker", "compose", "-f", str(compose_file(directory))]
    if observability:
        command.extend(["--profile", "observability"])
    command.extend(args)
    return command


def echo_result(result: CommandResult) -> None:
    if result.stdout.strip():
        typer.echo(result.stdout.rstrip())
    if result.stderr.strip():
        typer.echo(result.stderr.rstrip(), err=True)


def read_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def read_env(directory: Path) -> dict[str, str]:
    root = ensure_directory(directory)
    values = read_env_file(root / ".env.example")
    values.update(read_env_file(root / ".env"))
    return values


def api_url(directory: Path, explicit_url: str | None) -> str:
    if explicit_url:
        return explicit_url.rstrip("/")
    values = read_env(directory)
    host = values.get("API_HOST", "127.0.0.1")
    port = values.get("API_PORT", "8000")
    return f"http://{host}:{port}".rstrip("/")


def api_headers(directory: Path) -> dict[str, str]:
    api_key = read_env(directory).get("API_KEY", "")
    return {"x-api-key": api_key} if api_key else {}


def api_request(
    method: str,
    path: str,
    *,
    directory: Path,
    api_base_url: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> Any:
    url = f"{api_url(directory, api_base_url)}{path}"
    try:
        with httpx.Client(timeout=timeout, headers=api_headers(directory)) as client:
            response = client.request(method, url, json=payload)
    except httpx.HTTPError as exc:
        raise CliError(f"API request failed: {exc}", RUNTIME_FAILURE) from exc
    if response.status_code >= 400:
        raise CliError(f"API returned {response.status_code}: {response.text}", RUNTIME_FAILURE)
    if not response.text.strip():
        return None
    return response.json()


def print_json(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, sort_keys=True))


@app.command()
def init(
    directory: Annotated[Path, typer.Argument(help="Deployment directory to create.")],
    force: Annotated[bool, typer.Option("--force", help="Allow writing into a non-empty directory. Existing .env is preserved.")] = False,
) -> None:
    target = ensure_directory(directory)
    if target.exists() and any(target.iterdir()) and not force:
        fail(f"{target} is not empty. Re-run with --force to add templates.", INVALID_INPUT)
    target.mkdir(parents=True, exist_ok=True)
    with as_file(files("private_ai_stack").joinpath("templates", "deployment")) as template_root:
        for source in template_root.rglob("*"):
            relative = source.relative_to(template_root)
            destination = target / relative
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.name == ".env":
                continue
            if destination.exists() and destination.name == ".env":
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for dirname in ("audit", "reports", "exports", "backups"):
        (target / dirname).mkdir(parents=True, exist_ok=True)
    typer.echo(f"Initialized PrivateAIStack deployment template at {target}")


@config_app.command("check")
def config_check(
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    observability: Annotated[bool, typer.Option("--observability", help="Validate the observability profile too.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        root = ensure_directory(directory)
        required = ["compose.yaml", ".env.example", "config/otel-collector.yaml", "docker/postgres/init.sql"]
        missing = [item for item in required if not (root / item).exists()]
        if missing:
            raise CliError(f"Missing deployment files: {', '.join(missing)}", INVALID_INPUT)
        result = run_command(compose_args(root, "config", observability=observability), cwd=root)
        echo_result(result)
        if result.returncode != 0:
            raise CliError("Docker Compose configuration is invalid.", INVALID_INPUT)
        typer.echo("Configuration OK")
    except Exception as exc:
        handle_error(exc, verbose)


@app.command()
def up(
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    observability: Annotated[bool, typer.Option("--observability", help="Enable OpenTelemetry Collector and Jaeger.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        root = ensure_directory(directory)
        result = run_command(compose_args(root, "up", "--build", "-d", observability=observability), cwd=root, timeout=600)
        echo_result(result)
        if result.returncode != 0:
            raise CliError("Docker Compose startup failed.", RUNTIME_FAILURE)
    except Exception as exc:
        handle_error(exc, verbose)


@app.command()
def down(
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        root = ensure_directory(directory)
        result = run_command(compose_args(root, "down"), cwd=root, timeout=300)
        echo_result(result)
        if result.returncode != 0:
            raise CliError("Docker Compose shutdown failed.", RUNTIME_FAILURE)
    except Exception as exc:
        handle_error(exc, verbose)


@app.command()
def status(
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        root = ensure_directory(directory)
        result = run_command(compose_args(root, "ps"), cwd=root)
        echo_result(result)
        if result.returncode != 0:
            raise CliError("Could not read Docker Compose status.", RUNTIME_FAILURE)
    except Exception as exc:
        handle_error(exc, verbose)


@app.command()
def logs(
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    service: Annotated[str | None, typer.Option("--service", "-s", help="Optional Compose service name.")] = None,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Follow logs.")] = False,
    tail: Annotated[int, typer.Option("--tail", help="Number of log lines to show.", min=1)] = 100,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        root = ensure_directory(directory)
        args = ["logs", "--tail", str(tail)]
        if follow:
            args.append("-f")
        if service:
            args.append(service)
        result = stream_command(compose_args(root, *args), cwd=root) if follow else run_command(compose_args(root, *args), cwd=root)
        echo_result(result)
        if result.returncode != 0:
            raise CliError("Could not read Docker Compose logs.", RUNTIME_FAILURE)
    except Exception as exc:
        handle_error(exc, verbose)


@app.command()
def doctor(
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    observability: Annotated[bool, typer.Option("--observability", help="Include observability profile checks.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        root = ensure_directory(directory)
        rows: list[tuple[str, str, str]] = []
        rows.append(("python", "ok" if sys.version_info >= (3, 11) else "fail", sys.version.split()[0]))
        rows.append(("deployment directory", "ok" if root.exists() else "fail", str(root)))
        rows.append(("compose.yaml", "ok" if (root / "compose.yaml").exists() else "fail", str(root / "compose.yaml")))
        rows.append(
            (".env", "ok" if (root / ".env").exists() else "warn", "present" if (root / ".env").exists() else "using .env.example/defaults")
        )
        for dirname in ("audit", "reports", "exports", "backups"):
            path = root / dirname
            rows.append((dirname, "ok" if path.exists() else "warn", str(path)))
        docker = run_command(["docker", "--version"])
        rows.append(("docker cli", "ok" if docker.returncode == 0 else "fail", (docker.stdout or docker.stderr).strip()))
        compose = run_command(["docker", "compose", "version"])
        rows.append(("docker compose", "ok" if compose.returncode == 0 else "fail", (compose.stdout or compose.stderr).strip()))
        engine = run_command(["docker", "info"], timeout=30)
        rows.append(("docker engine", "ok" if engine.returncode == 0 else "warn", "available" if engine.returncode == 0 else "unavailable"))
        if (root / "compose.yaml").exists():
            config = run_command(compose_args(root, "config", observability=observability), cwd=root)
            rows.append(
                ("compose config", "ok" if config.returncode == 0 else "fail", "valid" if config.returncode == 0 else config.stderr.strip())
            )
        values = read_env(root)
        port_checks = (
            ("api port", int(values.get("API_PORT", "8000"))),
            ("ollama port", 11434),
            ("postgres port", 5432),
        )
        for name, port in port_checks:
            rows.append((name, "info", "open" if port_is_open("127.0.0.1", port) else "closed"))
        width = max(len(name) for name, _, _ in rows)
        for name, state, detail in rows:
            typer.echo(f"{name.ljust(width)}  {state.upper():<5}  {detail}")
        states = {state for _, state, _ in rows}
        if "fail" in states:
            raise typer.Exit(DEPENDENCY_UNAVAILABLE)
        if "warn" in states:
            raise typer.Exit(DEGRADED)
    except Exception as exc:
        handle_error(exc, verbose)


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


@app.command()
def health(
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    api_base_url: Annotated[str | None, typer.Option("--api-url", help="Override API URL.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        health_data = api_request("GET", "/health", directory=directory, api_base_url=api_base_url)
        ready_data = api_request("GET", "/ready", directory=directory, api_base_url=api_base_url)
        print_json({"health": health_data, "ready": ready_data})
        if isinstance(ready_data, dict) and ready_data.get("status") != "ready":
            raise typer.Exit(DEGRADED)
    except Exception as exc:
        handle_error(exc, verbose)


@model_app.command("list")
def model_list(
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        root = ensure_directory(directory)
        result = run_command(compose_args(root, "exec", "-T", "ollama", "ollama", "list"), cwd=root)
        echo_result(result)
        if result.returncode != 0:
            raise CliError("Could not list models in the Compose Ollama service.", RUNTIME_FAILURE)
    except Exception as exc:
        handle_error(exc, verbose)


@model_app.command("pull")
def model_pull(
    model: Annotated[str, typer.Option("--model", "-m", help="Model to pull into the Compose Ollama volume.")] = DEFAULT_MODEL,
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        root = ensure_directory(directory)
        result = run_command(compose_args(root, "exec", "-T", "ollama", "ollama", "pull", model), cwd=root, timeout=1800)
        echo_result(result)
        if result.returncode != 0:
            raise CliError(f"Could not pull model {model}.", RUNTIME_FAILURE)
    except Exception as exc:
        handle_error(exc, verbose)


@task_app.command("run")
def task_run(
    goal: Annotated[str, typer.Argument(help="Task goal for the local agent runtime.")],
    actor: Annotated[str, typer.Option("--actor", help="Audit actor label.")] = "local-user",
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    api_base_url: Annotated[str | None, typer.Option("--api-url", help="Override API URL.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        data = api_request("POST", "/v1/tasks", directory=directory, api_base_url=api_base_url, payload={"goal": goal, "actor": actor})
        print_json(data)
    except Exception as exc:
        handle_error(exc, verbose)


@knowledge_app.command("add")
def knowledge_add(
    content: Annotated[str | None, typer.Option("--content", help="Inline document content.")] = None,
    file: Annotated[Path | None, typer.Option("--file", help="File to ingest.")] = None,
    source_name: Annotated[str | None, typer.Option("--source-name", help="Source label.")] = None,
    replace_existing: Annotated[bool, typer.Option("--replace-existing", help="Replace matching content.")] = False,
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    api_base_url: Annotated[str | None, typer.Option("--api-url", help="Override API URL.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        if bool(content) == bool(file):
            raise CliError("Provide exactly one of --content or --file.", INVALID_INPUT)
        body = content if content is not None else file.read_text(encoding="utf-8") if file is not None else ""
        source = source_name or (str(file) if file is not None else "inline-document")
        payload = {"content": body, "source_name": source, "replace_existing": replace_existing, "metadata": {}}
        data = api_request("POST", "/v1/knowledge/documents", directory=directory, api_base_url=api_base_url, payload=payload)
        print_json(data)
    except Exception as exc:
        handle_error(exc, verbose)


@knowledge_app.command("search")
def knowledge_search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    limit: Annotated[int, typer.Option("--limit", help="Maximum hits.", min=1, max=20)] = 5,
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    api_base_url: Annotated[str | None, typer.Option("--api-url", help="Override API URL.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        data = api_request(
            "POST", "/v1/knowledge/search", directory=directory, api_base_url=api_base_url, payload={"query": query, "limit": limit}
        )
        print_json(data)
    except Exception as exc:
        handle_error(exc, verbose)


@app.command()
def review(
    repository_path: Annotated[str, typer.Argument(help="Repository path visible to the API container.")],
    mode: Annotated[str, typer.Option("--mode", help="Review mode.")] = "safe-static",
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    api_base_url: Annotated[str | None, typer.Option("--api-url", help="Override API URL.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        payload = {"repository_path": str(repository_path), "mode": mode}
        data = api_request("POST", "/v1/reviews", directory=directory, api_base_url=api_base_url, payload=payload)
        print_json(data)
    except Exception as exc:
        handle_error(exc, verbose)


@audit_app.command("show")
def audit_show(
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    lines: Annotated[int, typer.Option("--lines", "-n", help="Number of audit records to show.", min=0)] = 20,
) -> None:
    path = ensure_directory(directory) / "audit" / "audit.jsonl"
    if not path.exists():
        fail(f"No audit log found at {path}", INVALID_INPUT)
    records: deque[str] = deque(maxlen=lines)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(line.rstrip())
    for line in records:
        typer.echo(line)


@audit_app.command("export")
def audit_export(
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output JSONL path.")] = None,
) -> None:
    root = ensure_directory(directory)
    source = root / "audit" / "audit.jsonl"
    if not source.exists():
        fail(f"No audit log found at {source}", INVALID_INPUT)
    destination = output or root / "exports" / f"audit-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    typer.echo(str(destination.resolve()))


@audit_app.command("verify")
def audit_verify(directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path(".")) -> None:
    path = ensure_directory(directory) / "audit" / "audit.jsonl"
    if not path.exists():
        fail(f"No audit log found at {path}", INVALID_INPUT)
    valid, count, reason = AuditWriter(path, verify_existing=False).verify()
    if not valid:
        fail(reason or "Audit chain verification failed.", RUNTIME_FAILURE)
    typer.echo(f"Audit chain OK ({count} records)")


@app.command()
def backup(
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Optional backup output path.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        root = ensure_directory(directory)
        destination = output or root / "backups" / f"private_ai_stack_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.sql"
        destination.parent.mkdir(parents=True, exist_ok=True)
        dump = run_command(
            compose_args(root, "exec", "-T", "postgres", "pg_dump", "-U", "private_ai_stack", "private_ai_stack"), cwd=root, timeout=600
        )
        if dump.returncode != 0:
            echo_result(dump)
            raise CliError("PostgreSQL backup failed.", RUNTIME_FAILURE)
        destination.write_text(dump.stdout, encoding="utf-8")
        typer.echo(str(destination.resolve()))
    except Exception as exc:
        handle_error(exc, verbose)


@app.command()
def restore(
    backup_file: Annotated[Path, typer.Argument(help="SQL backup file to restore.")],
    directory: Annotated[Path, typer.Option("--directory", "-d", help="Deployment directory.")] = Path("."),
    verbose: Annotated[bool, typer.Option("--verbose", help="Show raw exceptions.")] = False,
) -> None:
    try:
        root = ensure_directory(directory)
        source = backup_file.expanduser().resolve()
        if not source.exists():
            raise CliError(f"Backup file not found: {source}", INVALID_INPUT)
        ps = run_command_with_file_stdin(
            compose_args(root, "exec", "-T", "postgres", "psql", "-U", "private_ai_stack", "private_ai_stack"),
            source,
            cwd=root,
            timeout=600,
        )
        if ps.returncode != 0:
            echo_result(ps)
            raise CliError("PostgreSQL restore failed.", RUNTIME_FAILURE)
        typer.echo("Restore completed")
    except Exception as exc:
        handle_error(exc, verbose)


if __name__ == "__main__":
    main()
