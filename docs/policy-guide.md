# Policy Guide

PrivateAIStack uses Forge policy-as-code where compatible and a narrow local policy catalog for documented v0.1 behavior.

Default policies:

- Deny `.env`, private-key, SSH, and credential-file reads.
- Require approval before external network access.
- Log database mutations.

Missing approval infrastructure denies by default. Console approval is intended only for local interactive development.
