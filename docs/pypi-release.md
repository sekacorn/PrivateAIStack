# PyPI Release

The distribution name is `privateaistack`; it already exists on PyPI and must be published through the existing project.

Do not publish from routine development or CI. CI builds artifacts and checks metadata only. Release publication uses GitHub Actions with PyPI Trusted Publishing.

## Trusted Publishing Setup

In GitHub, create or verify an environment named exactly `pypi`.

In PyPI, open the existing `privateaistack` project and configure a GitHub Actions Trusted Publisher:

- PyPI project: `privateaistack`
- GitHub owner: `sekacorn`
- GitHub repository: `PrivateAIStack`
- Workflow filename: `release.yml`
- Environment: `pypi`

Do not add a PyPI API token secret.

## Release

```bash
git tag -a v0.2.0b1 -m "privateaistack 0.2.0b1"
git push origin v0.2.0b1
```

Only tag after local checks, artifact inspection, clean wheel and source-distribution installs, Compose validation, pushed CI, GitHub environment setup, and PyPI Trusted Publisher configuration are complete.
