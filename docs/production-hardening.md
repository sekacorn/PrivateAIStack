# Production Hardening

PrivateAIStack `0.2.0b1` is a Beta candidate and is not production-hardened by default.

Before production use:

- Review container isolation and runtime privileges.
- Replace development passwords.
- Configure authentication and authorization.
- Review logging, telemetry, audit retention, and redaction.
- Perform threat modeling for the deployment environment.
- Validate accessibility, records, privacy, and security requirements.
- Pin all images and actions for the release.
- Run dependency, container, and secret scans.
- Decide whether stronger isolation such as VMs or microVMs is required.
