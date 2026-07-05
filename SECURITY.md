# Security Policy

Report security concerns privately to Sekacorn@gmail.com.

Do not open public issues for suspected vulnerabilities that include exploit details, credentials, private code, or sensitive operational information.

PrivateAIStack is demonstration-grade v0.1 software. It provides application-level controls for local AI workflows, but it is not VM-grade isolation and does not claim regulatory compliance.

Security-sensitive defaults:

- Hosted providers are disabled by default.
- Ollama is the default local model provider.
- PostgreSQL is not published to the host by default.
- The Docker socket is not mounted.
- Missing approvers deny by default.
- Reviewed repositories are handled in `safe-static` mode by default.
