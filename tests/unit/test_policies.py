from private_ai_stack.policies.defaults import build_policy_set, policy_catalog


def test_policy_catalog_documents_default_outcomes() -> None:
    catalog = policy_catalog()
    assert any(item.policy == "deny-sensitive-file-read" for item in catalog)
    assert any(item.action == "approve" and not item.allowed for item in catalog)


async def test_sensitive_path_policy_denies() -> None:
    policy = build_policy_set()
    decision = await policy.evaluate("read_file", {"path": "/workspace/target/.env"})
    assert decision.allowed is False
    assert decision.action_taken == "denied"
