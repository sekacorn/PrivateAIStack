# Security Policy

Use GitHub Security Advisories if enabled, or contact the maintainer through the repository before sharing sensitive vulnerability details.

Do not open public issues for suspected vulnerabilities that include exploit details, credentials, private code, or sensitive operational information.

PrivateAIStack is Beta-candidate software. It provides application-level controls for local AI workflows, but it is not production-hardened, VM-grade isolation, or a claim of regulatory compliance.

Security-sensitive defaults:

- Hosted providers are disabled by default.
- Ollama is the default local model provider.
- PostgreSQL is not published to the host by default.
- The Docker socket is not mounted.
- Missing approvers deny by default.
- Reviewed repositories are handled in `safe-static` mode by default.
