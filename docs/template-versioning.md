# Template Versioning

Deployment templates live in `private_ai_stack/templates/deployment/` and are copied with `importlib.resources`.

Each template includes `.privateaistack-template.json` with:

- Template name.
- Template version.
- Default model.
- Default runtime profile.
- Pinned Ollama image.
- Observability profile name.

The template version should normally match the Python package version for alpha releases. If a future patch changes only the template and not runtime code, document the compatibility boundary here and in `CHANGELOG.md`.

Template rules:

- Never include real secrets.
- Never overwrite an existing `.env`.
- Keep default bindings on localhost.
- Preserve named Docker volumes.
- Keep observability optional through the `observability` Compose profile.
- Do not automatically pull large models.
