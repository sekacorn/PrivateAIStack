"""The AgentForge conformance claim must hold for the whole supported range.

``test_reembed_embedders.py`` holds the reimplemented AgentForge scheme to
AgentForge's own output, but only for whichever ``agentforge-oss`` happens to
be installed. That is the weaker claim: a resolver could pin the floor forever
and the conformance test would keep passing while a newer release in the
supported range quietly changed the scheme.

So the claim is split in two:

* Always, in any environment: the installed ``agentforge-oss`` satisfies the
  range this project declares. A stale pin or a resolver surprise fails here.
* In the ``embedder-conformance`` CI job: the installed version is the *latest*
  release in that range. The job resolves it with pip, exports it as
  ``PRIVATEAISTACK_AGENTFORGE_EXPECTED``, and the conformance tests then run
  against it.

Resolving the latest release needs the index, and tests here are offline, which
is why the resolution happens in the workflow and this module only checks what
it was handed. Nothing here contacts PyPI.
"""

from __future__ import annotations

import importlib.metadata
import os
import tomllib
from pathlib import Path
from typing import Final

import pytest

PYPROJECT: Final = Path(__file__).resolve().parents[2] / "pyproject.toml"
DISTRIBUTION: Final = "agentforge-oss"
EXPECTED_ENV: Final = "PRIVATEAISTACK_AGENTFORGE_EXPECTED"


def _release(version: str) -> tuple[int, ...]:
    """The numeric release segment of ``version``, for ordering.

    Deliberately small: these constraints are plain ``>=`` / ``<`` bounds on
    simple versions. A suffix such as ``b1`` stops the parse, so ``0.6.0b1``
    orders as ``0.6.0`` and is treated as inside a ``<0.6.0`` bound only if
    the numeric parts allow it. If a real prerelease ever has to be supported,
    this test must be rewritten rather than stretched.
    """
    parts: list[int] = []
    for segment in version.split("."):
        digits = ""
        for character in segment:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        raise AssertionError(f"Cannot order version {version!r}.")
    return tuple(parts)


def _declared_constraint() -> str:
    """The bare ``agentforge-oss`` requirement from ``project.dependencies``."""
    document = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    matches = [
        requirement
        for requirement in document["project"]["dependencies"]
        if requirement.split("[")[0].split(">")[0].split("<")[0].split("=")[0].strip() == DISTRIBUTION and "[" not in requirement
    ]
    assert len(matches) == 1, f"Expected exactly one bare {DISTRIBUTION} dependency, found {matches}."
    return matches[0]


def _bounds(constraint: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    specifier = constraint[len(DISTRIBUTION) :]
    lower: tuple[int, ...] | None = None
    upper: tuple[int, ...] | None = None
    for clause in specifier.split(","):
        clause = clause.strip()
        if clause.startswith(">="):
            lower = _release(clause[2:])
        elif clause.startswith("<") and not clause.startswith("<="):
            upper = _release(clause[1:])
        else:
            raise AssertionError(f"Unsupported clause {clause!r}; this test only understands >= and <.")
    assert lower is not None and upper is not None, f"{constraint!r} must have both a floor and a ceiling."
    return lower, upper


def test_the_declared_constraint_is_a_bounded_range() -> None:
    """An unbounded dependency would make 'the supported range' meaningless."""
    lower, upper = _bounds(_declared_constraint())
    assert lower < upper


def test_the_installed_agentforge_satisfies_the_declared_range() -> None:
    installed = _release(importlib.metadata.version(DISTRIBUTION))
    lower, upper = _bounds(_declared_constraint())
    assert lower <= installed < upper, f"Installed {installed} is outside {_declared_constraint()!r}."


def test_the_conformance_run_used_the_latest_release_in_the_range() -> None:
    """Only the ``embedder-conformance`` job sets the expectation; elsewhere this skips."""
    expected = os.environ.get(EXPECTED_ENV)
    if not expected:
        pytest.skip(f"{EXPECTED_ENV} is unset; the latest-in-range check runs in CI.")
    installed = importlib.metadata.version(DISTRIBUTION)
    assert installed == expected, f"Job resolved {expected}, but {installed} is installed."
    lower, upper = _bounds(_declared_constraint())
    assert lower <= _release(installed) < upper
