# Security Policy

Report a suspected vulnerability privately with **Report a vulnerability** on the repository's
[Security tab](https://github.com/sekacorn/PrivateAIStack/security) (direct link: [github.com/sekacorn/PrivateAIStack/security/advisories/new](https://github.com/sekacorn/PrivateAIStack/security/advisories/new)).
This is GitHub's private vulnerability reporting: the report stays private between you and the
maintainers until an advisory is published.

## Published advisories

| Advisory | Affected | Fixed in |
|---|---|---|
| [GHSA-mhfj-hqw5-84h5](https://github.com/sekacorn/PrivateAIStack/security/advisories/GHSA-mhfj-hqw5-84h5): URL settings accepted credentials and query-string tokens that could reach INFO logs | `<= 0.2.0b2` | `0.2.0b3` |

All advisories: [github.com/sekacorn/PrivateAIStack/security/advisories](https://github.com/sekacorn/PrivateAIStack/security/advisories).

Do not open public issues for suspected vulnerabilities that include exploit details, credentials, private code, or sensitive operational information.

PrivateAIStack is Beta-candidate software. It provides application-level controls for local AI workflows, but it is not production-hardened, VM-grade isolation, or a claim of regulatory compliance.

Security-sensitive defaults:

- Hosted providers are disabled by default.
- Ollama is the default local model provider.
- PostgreSQL is not published to the host by default.
- The Docker socket is not mounted.
- Missing approvers deny by default.
- Reviewed repositories are handled in `safe-static` mode by default.
