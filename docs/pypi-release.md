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

The release workflow runs on a version tag push, or on `workflow_dispatch`
with a `tag` and a `dry_run` input that defaults to `true`. A dry run does the
tag/version check, lint, type check, Bandit, tests, build and `twine check`,
then stops before the upload. **A pushed version tag is always a real
release**, so rehearse first against an existing tag:

```bash
gh workflow run release.yml --repo sekacorn/PrivateAIStack -f tag=v0.2.0b3
```

Then tag the verified commit and push the tag by explicit refspec, never with
`--tags`:

```bash
git tag -a v0.2.0b4 <verified-commit> -m "privateaistack 0.2.0b4"
git push origin refs/tags/v0.2.0b4:refs/tags/v0.2.0b4
```

The publish job waits at the `pypi` environment for a required reviewer to
approve it. A published version can never be re-uploaded.

Only tag after local checks, artifact inspection, clean wheel and source-distribution installs, Compose validation, pushed CI, GitHub environment setup, and PyPI Trusted Publisher configuration are complete.
