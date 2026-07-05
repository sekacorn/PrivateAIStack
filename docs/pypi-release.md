# PyPI Release

The preferred distribution name is `privateaistack`. It was checked against PyPI's JSON API during packaging work and appeared available then, but the name is not reserved until an upload succeeds.

Do not publish from routine development or CI. CI builds artifacts and checks metadata only.

## Manual TestPyPI Upload

```bash
python -m pip install --upgrade build twine
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
```

Then test in a clean environment:

```bash
python -m venv .testpypi-venv
.testpypi-venv/Scripts/python -m pip install --upgrade pip
.testpypi-venv/Scripts/python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ privateaistack==0.1.0a1
.testpypi-venv/Scripts/privateaistack --version
```

## Manual PyPI Upload

Publish only after TestPyPI install, local quality checks, and artifact inspection pass:

```bash
python -m twine upload dist/*
```

Use a scoped PyPI token or trusted publishing. Do not place tokens in `.env`, shell history, the repository, or CI logs.
