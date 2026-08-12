from __future__ import annotations

import json
from importlib.resources import files

from typer.testing import CliRunner

from private_ai_stack import __version__
from private_ai_stack.audit.writer import AuditWriter
from private_ai_stack.cli import main as cli

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert f"privateaistack {__version__}" in result.stdout


def test_deployment_template_is_packaged() -> None:
    template_root = files("private_ai_stack").joinpath("templates", "deployment")

    assert template_root.joinpath("compose.yaml").is_file()
    assert template_root.joinpath("Dockerfile").is_file()
    assert template_root.joinpath(".env.example").is_file()
    assert template_root.joinpath("config", "otel-collector.yaml").is_file()
    assert template_root.joinpath("docker", "postgres", "init.sql").is_file()
    template_metadata = json.loads(template_root.joinpath(".privateaistack-template.json").read_text(encoding="utf-8"))
    assert template_metadata["version"] == __version__


def test_init_refuses_non_empty_directory(tmp_path) -> None:
    target = tmp_path / "deployment"
    target.mkdir()
    (target / "keep.txt").write_text("user file", encoding="utf-8")

    result = runner.invoke(cli.app, ["init", str(target)])

    assert result.exit_code == cli.INVALID_INPUT
    assert "not empty" in result.output


def test_init_force_preserves_env(tmp_path) -> None:
    target = tmp_path / "deployment"
    target.mkdir()
    (target / ".env").write_text("API_KEY=user-secret\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["init", str(target), "--force"])

    assert result.exit_code == 0
    assert (target / ".env").read_text(encoding="utf-8") == "API_KEY=user-secret\n"
    assert (target / "compose.yaml").exists()
    assert (target / "README-DEPLOYMENT.md").exists()
    assert (target / "audit").is_dir()


def test_config_check_runs_compose_config(tmp_path, monkeypatch) -> None:
    target = tmp_path / "deployment"
    runner.invoke(cli.app, ["init", str(target)])
    seen_args: list[list[str]] = []

    def fake_run(args: list[str], cwd=None, timeout=120, stdin=None):  # noqa: ANN001
        seen_args.append(args)
        return cli.CommandResult(args=args, returncode=0, stdout="services: {}\n", stderr="")

    monkeypatch.setattr(cli, "run_command", fake_run)

    result = runner.invoke(cli.app, ["config", "check", "--directory", str(target)])

    assert result.exit_code == 0
    assert seen_args == [["docker", "compose", "-f", str(target.resolve() / "compose.yaml"), "config"]]


def test_down_does_not_remove_volumes(tmp_path, monkeypatch) -> None:
    target = tmp_path / "deployment"
    runner.invoke(cli.app, ["init", str(target)])
    seen_args: list[list[str]] = []

    def fake_run(args: list[str], cwd=None, timeout=120, stdin=None):  # noqa: ANN001
        seen_args.append(args)
        return cli.CommandResult(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli, "run_command", fake_run)

    result = runner.invoke(cli.app, ["down", "--directory", str(target)])

    assert result.exit_code == 0
    assert seen_args == [["docker", "compose", "-f", str(target.resolve() / "compose.yaml"), "down"]]
    assert "-v" not in seen_args[0]


def test_compose_management_commands(tmp_path, monkeypatch) -> None:
    target = tmp_path / "deployment"
    runner.invoke(cli.app, ["init", str(target)])
    seen_args: list[list[str]] = []

    def fake_run(args: list[str], cwd=None, timeout=120, stdin=None):  # noqa: ANN001
        seen_args.append(args)
        return cli.CommandResult(args=args, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(cli, "run_command", fake_run)

    commands = [
        ["up", "--directory", str(target), "--observability"],
        ["status", "--directory", str(target)],
        ["logs", "--directory", str(target), "--service", "api"],
        ["model", "list", "--directory", str(target)],
        ["model", "pull", "--directory", str(target), "--model", "qwen2.5:3b"],
    ]

    for command in commands:
        result = runner.invoke(cli.app, command)
        assert result.exit_code == 0

    assert [
        "docker",
        "compose",
        "-f",
        str(target.resolve() / "compose.yaml"),
        "--profile",
        "observability",
        "up",
        "--build",
        "-d",
    ] in seen_args
    assert ["docker", "compose", "-f", str(target.resolve() / "compose.yaml"), "ps"] in seen_args
    assert ["docker", "compose", "-f", str(target.resolve() / "compose.yaml"), "logs", "--tail", "100", "api"] in seen_args
    assert ["docker", "compose", "-f", str(target.resolve() / "compose.yaml"), "exec", "-T", "ollama", "ollama", "list"] in seen_args
    assert [
        "docker",
        "compose",
        "-f",
        str(target.resolve() / "compose.yaml"),
        "exec",
        "-T",
        "ollama",
        "ollama",
        "pull",
        "qwen2.5:3b",
    ] in seen_args


def test_logs_follow_streams_without_capture(tmp_path, monkeypatch) -> None:
    target = tmp_path / "deployment"
    runner.invoke(cli.app, ["init", str(target)])
    seen_args: list[list[str]] = []

    def fake_stream(args: list[str], cwd=None):  # noqa: ANN001
        seen_args.append(args)
        return cli.CommandResult(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli, "stream_command", fake_stream)

    result = runner.invoke(cli.app, ["logs", "--directory", str(target), "--follow", "--service", "api"])

    assert result.exit_code == 0
    assert seen_args == [["docker", "compose", "-f", str(target.resolve() / "compose.yaml"), "logs", "--tail", "100", "-f", "api"]]


def test_doctor_reports_degraded_without_env(tmp_path, monkeypatch) -> None:
    target = tmp_path / "deployment"
    runner.invoke(cli.app, ["init", str(target)])

    def fake_run(args: list[str], cwd=None, timeout=120, stdin=None):  # noqa: ANN001
        if args == ["docker", "--version"]:
            return cli.CommandResult(args=args, returncode=0, stdout="Docker version test\n", stderr="")
        if args == ["docker", "compose", "version"]:
            return cli.CommandResult(args=args, returncode=0, stdout="Docker Compose version test\n", stderr="")
        if args == ["docker", "info"]:
            return cli.CommandResult(args=args, returncode=0, stdout="engine\n", stderr="")
        return cli.CommandResult(args=args, returncode=0, stdout="services: {}\n", stderr="")

    monkeypatch.setattr(cli, "run_command", fake_run)
    checked_ports: list[int] = []

    def fake_port_is_open(host: str, port: int) -> bool:
        checked_ports.append(port)
        return False

    monkeypatch.setattr(cli, "port_is_open", fake_port_is_open)

    result = runner.invoke(cli.app, ["doctor", "--directory", str(target)])

    assert result.exit_code == cli.DEGRADED
    assert "docker cli" in result.stdout
    assert "using .env.example/defaults" in result.stdout
    assert 8000 in checked_ports


def test_health_ready_success(monkeypatch, tmp_path) -> None:
    def fake_api(method: str, path: str, **kwargs):  # noqa: ANN001
        if path == "/health":
            return {"status": "ok"}
        return {"status": "ready", "checks": {"ollama": "ok"}}

    monkeypatch.setattr(cli, "api_request", fake_api)

    result = runner.invoke(cli.app, ["health", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    assert '"ready"' in result.stdout


def test_health_degraded(monkeypatch, tmp_path) -> None:
    def fake_api(method: str, path: str, **kwargs):  # noqa: ANN001
        if path == "/health":
            return {"status": "ok"}
        return {"status": "degraded", "checks": {"ollama": "unavailable"}}

    monkeypatch.setattr(cli, "api_request", fake_api)

    result = runner.invoke(cli.app, ["health", "--directory", str(tmp_path)])

    assert result.exit_code == cli.DEGRADED


def test_api_backed_commands(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def fake_api(method: str, path: str, **kwargs):  # noqa: ANN001
        calls.append((method, path, kwargs.get("payload")))
        return {"path": path, "ok": True}

    monkeypatch.setattr(cli, "api_request", fake_api)
    source_file = tmp_path / "source.md"
    source_file.write_text("knowledge text", encoding="utf-8")

    commands = [
        ["task", "run", "--directory", str(tmp_path), "Do the thing"],
        ["knowledge", "add", "--directory", str(tmp_path), "--file", str(source_file)],
        ["knowledge", "search", "--directory", str(tmp_path), "knowledge"],
        ["review", "--directory", str(tmp_path), "/app/sample-target"],
    ]

    for command in commands:
        result = runner.invoke(cli.app, command)
        assert result.exit_code == 0

    assert ("POST", "/v1/tasks", {"goal": "Do the thing", "actor": "local-user"}) in calls
    assert calls[1][0:2] == ("POST", "/v1/knowledge/documents")
    assert calls[2] == ("POST", "/v1/knowledge/search", {"query": "knowledge", "limit": 5})
    assert calls[3] == ("POST", "/v1/reviews", {"repository_path": "/app/sample-target", "mode": "safe-static"})


def test_knowledge_add_requires_one_source(tmp_path) -> None:
    result = runner.invoke(cli.app, ["knowledge", "add", "--directory", str(tmp_path)])

    assert result.exit_code == cli.INVALID_INPUT
    assert "exactly one" in result.output


def test_config_check_reports_missing_files(tmp_path) -> None:
    result = runner.invoke(cli.app, ["config", "check", "--directory", str(tmp_path)])

    assert result.exit_code == cli.INVALID_INPUT
    assert "Missing deployment files" in result.output


def test_env_helpers_use_directory_env(tmp_path) -> None:
    (tmp_path / ".env.example").write_text("API_HOST=127.0.0.1\nAPI_PORT=8000\nAPI_KEY=\nDEFAULT_PROFILE=laptop-cpu\n", encoding="utf-8")
    (tmp_path / ".env").write_text("API_HOST=127.0.0.2\nAPI_PORT=9000\nAPI_KEY=secret\n", encoding="utf-8")

    assert cli.read_env(tmp_path)["API_KEY"] == "secret"
    assert cli.read_env(tmp_path)["DEFAULT_PROFILE"] == "laptop-cpu"
    assert cli.api_url(tmp_path, None) == "http://127.0.0.2:9000"
    assert cli.api_headers(tmp_path) == {"x-api-key": "secret"}
    assert cli.api_url(tmp_path, "http://localhost:8080/") == "http://localhost:8080"


def test_run_command_reports_missing_dependency(monkeypatch) -> None:
    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise FileNotFoundError("missing")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    try:
        cli.run_command(["missing-command"])
    except cli.CliError as exc:
        assert exc.exit_code == cli.DEPENDENCY_UNAVAILABLE
    else:
        raise AssertionError("expected CliError")


def test_api_request_success_and_error(monkeypatch, tmp_path) -> None:
    class FakeResponse:
        status_code = 200
        text = '{"ok": true}'

        def json(self):  # noqa: ANN201
            return {"ok": True}

    class FakeClient:
        def __init__(self, timeout, headers):  # noqa: ANN001
            self.timeout = timeout
            self.headers = headers

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204
            return False

        def request(self, method, url, json=None):  # noqa: ANN001, A002, ANN201
            assert method == "GET"
            assert url.endswith("/health")
            return FakeResponse()

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    assert cli.api_request("GET", "/health", directory=tmp_path) == {"ok": True}


def test_audit_show_and_export(tmp_path) -> None:
    audit_path = tmp_path / "audit" / "audit.jsonl"
    audit_path.parent.mkdir()
    audit_path.write_text('{"event_type":"one"}\n{"event_type":"two"}\n', encoding="utf-8")

    show = runner.invoke(cli.app, ["audit", "show", "--directory", str(tmp_path), "--lines", "1"])
    export = runner.invoke(cli.app, ["audit", "export", "--directory", str(tmp_path), "--output", str(tmp_path / "out.jsonl")])

    assert show.exit_code == 0
    assert '"two"' in show.stdout
    assert export.exit_code == 0
    assert (tmp_path / "out.jsonl").read_text(encoding="utf-8") == audit_path.read_text(encoding="utf-8")


def test_backup_and_restore(tmp_path, monkeypatch) -> None:
    target = tmp_path / "deployment"
    runner.invoke(cli.app, ["init", str(target)])
    backup_file = tmp_path / "backup.sql"
    backup_file.write_text("select 1;\n", encoding="utf-8")
    seen_restore_file: list[str] = []

    def fake_run(args: list[str], cwd=None, timeout=120, stdin=None):  # noqa: ANN001
        stdout = "dump\n" if "pg_dump" in args else ""
        return cli.CommandResult(args=args, returncode=0, stdout=stdout, stderr="")

    def fake_run_with_file(args: list[str], input_path, cwd=None, timeout=120):  # noqa: ANN001
        seen_restore_file.append(input_path.read_text(encoding="utf-8"))
        return cli.CommandResult(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli, "run_command", fake_run)
    monkeypatch.setattr(cli, "run_command_with_file_stdin", fake_run_with_file)

    backup = runner.invoke(cli.app, ["backup", "--directory", str(target), "--output", str(tmp_path / "dump.sql")])
    restore = runner.invoke(cli.app, ["restore", str(backup_file), "--directory", str(target)])

    assert backup.exit_code == 0
    assert (tmp_path / "dump.sql").read_text(encoding="utf-8") == "dump\n"
    assert restore.exit_code == 0
    assert seen_restore_file == ["select 1;\n"]


def test_audit_verify_detects_valid_chain(tmp_path) -> None:
    audit_path = tmp_path / "audit" / "audit.jsonl"
    writer = AuditWriter(audit_path)
    writer.write("task.created", entity_type="task", entity_id="task-1", details={"goal": "test"})
    writer.write("task.completed", entity_type="task", entity_id="task-1", details={"status": "ok"})

    result = runner.invoke(cli.app, ["audit", "verify", "--directory", str(tmp_path)])

    assert result.exit_code == 0
    assert "Audit chain OK (2 records)" in result.stdout


def test_audit_verify_detects_tampering(tmp_path) -> None:
    audit_path = tmp_path / "audit" / "audit.jsonl"
    writer = AuditWriter(audit_path)
    writer.write("task.created", entity_type="task", entity_id="task-1", details={"goal": "test"})
    record = json.loads(audit_path.read_text(encoding="utf-8"))
    record["details"]["goal"] = "tampered"
    audit_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["audit", "verify", "--directory", str(tmp_path)])

    assert result.exit_code == cli.RUNTIME_FAILURE
    assert "hash mismatch" in result.output
