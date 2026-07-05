from typing import Any

from forge import PolicyRule, PolicySet

from private_ai_stack.policies.models import PolicyOutcome


def _contains_sensitive_path(_: str, args: dict[str, Any]) -> bool:
    path = str(args.get("path", "")).lower()
    denied = [".env", ".ssh", "id_rsa", "id_ed25519", "credentials", "private.key", "secret"]
    return any(item in path for item in denied)


def _is_network(_: str, args: dict[str, Any]) -> bool:
    return bool(args.get("url") or args.get("host"))


async def _deny_approver(_: str, __: dict[str, Any]) -> bool:
    return False


def build_policy_set() -> PolicySet:
    policy = PolicySet()
    policy.add(
        PolicyRule(
            name="deny-sensitive-file-read",
            description="Deny .env, private key, SSH, and credential-file reads.",
            tool_names=["read_file", "cat", "open"],
            condition=_contains_sensitive_path,
            action="deny",
        )
    )
    policy.add(
        PolicyRule(
            name="require-approval-for-network",
            description="External network access requires approval and denies by default without an approver.",
            tool_names=["http_get", "network", "curl"],
            condition=_is_network,
            action="approve",
            approver=_deny_approver,
        )
    )
    policy.add(
        PolicyRule(
            name="log-database-mutation",
            description="Database mutations are logged for auditability.",
            tool_names=["database_write", "sql_execute"],
            condition=lambda _name, args: (
                str(args.get("sql", "")).strip().lower().startswith(("insert", "update", "delete", "alter", "drop"))
            ),
            action="log",
        )
    )
    return policy


def policy_catalog() -> list[PolicyOutcome]:
    return [
        PolicyOutcome(
            policy="deny-sensitive-file-read",
            action="deny",
            allowed=False,
            reason="Deny .env, private-key, SSH, and credential-file reads.",
        ),
        PolicyOutcome(
            policy="require-approval-for-network",
            action="approve",
            allowed=False,
            reason="Requires approval; denies by default when no approver is available.",
        ),
        PolicyOutcome(
            policy="log-database-mutation",
            action="log",
            allowed=True,
            reason="Database mutations are allowed only when the caller is authorized and are audited.",
        ),
    ]
