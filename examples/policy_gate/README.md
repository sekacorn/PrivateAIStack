# Policy Gate Example

List configured policy behavior:

```bash
curl -sS http://127.0.0.1:8000/v1/policies
```

Default v0.1 policies:

- deny sensitive file reads;
- require approval for external network access;
- log database mutations.

When no approver is available, approval-required actions deny by default.
