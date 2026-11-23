"""The egress gate in front of a re-embedding run.

Checkpoint 3b-5. Re-embedding sends source text to a model, which the 3a
evidence work never did. That is the sharpest new risk in the phase, so it gets
a gate rather than a convention.

Two things are true of this module and nothing else in the package makes them
true. A run is local unless an operator explicitly says otherwise, and a run
that sent anything says so in a record it cannot omit. Both halves matter: a
default nobody can trip over, and a result nobody can quietly misreport.

**This package ships no hosted provider.** It ships the gate, the seam an
operator plugs one into, and the record. Choosing a vendor, writing its client
and adding a runtime dependency on an HTTP library are decisions that belong to
whoever is doing the sending, not to the tool that refuses to do it by default.
A hosted run therefore needs an operator-supplied embedder, and the pipeline
says so rather than silently falling back to a local one -- a fallback would
turn "my hosted run worked" into "something ran, locally, and the record says
hosted".

Why the warnings are restated rather than imported
--------------------------------------------------

ModelSwapBench's ``provider_data_preflight`` already decides this question, and
PrivateAIStack and ModelSwapBench deliberately take no dependency on each
other. So the shape and the warning sentences are restated here, verbatim, and
a test pins the strings: an operator who has read one tool's warning should not
have to learn a second wording for the same risk. If the two ever drift, the
test is where it shows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

LOCAL: Final = "local"
HOSTED: Final = "hosted"
HOSTING_MODES: Final = frozenset({LOCAL, HOSTED})

PRIVACY_CLASSIFICATIONS: Final = frozenset({"public", "internal", "confidential", "restricted", "unknown"})
DEFAULT_CLASSIFICATION: Final = "unknown"
"""``unknown`` rather than ``public``.

A default of ``public`` would assert something about an operator's data that
nobody said, and it is the one classification that silences the unredacted
warning. ``unknown`` keeps the warning.
"""

OPEN_CLASSIFICATIONS: Final = frozenset({"public", DEFAULT_CLASSIFICATION})
"""Classifications that do not raise the unredacted-content warning.

``unknown`` is here because ModelSwapBench puts it here, and the two wordings
must not diverge. It is the weaker of the two choices and is called out in the
documentation rather than quietly changed.
"""

BLOCKED_WARNING: Final = "hosted execution is blocked until explicitly enabled"
UNREDACTED_WARNING: Final = "private unredacted dataset content would be transmitted to a hosted provider"
TELEMETRY_WARNING: Final = "telemetry is enabled; review exported fields before execution"
MAX_WARNINGS: Final = 16
MAX_PROVIDER_CHARS: Final = 128


class EgressError(ValueError):
    """Raised when a run's egress settings are not usable as declared."""


@dataclass(frozen=True)
class EgressPreflight:
    """What this run is allowed to send, and what the operator was told.

    ``allowed`` is the gate. ``content_leaves_machine`` is the consequence: a
    hosted run that was allowed. They are separate because a hosted run can be
    allowed and still send nothing, and because a record has to be able to say
    that an attempt was blocked rather than never made.
    """

    provider: str | None
    hosting_mode: str
    dataset_privacy_classification: str
    hosted_execution_confirmed: bool
    redaction_applied: bool
    telemetry_enabled: bool
    allowed: bool
    content_leaves_machine: bool
    warnings: tuple[str, ...]

    @property
    def is_hosted(self) -> bool:
        return self.hosting_mode == HOSTED

    @property
    def blocked(self) -> bool:
        """A hosted run nobody enabled."""
        return self.is_hosted and not self.allowed

    def to_document(self, *, records_transmitted: int) -> dict[str, Any]:
        """The ``egress`` block of a ``re-embed-record.v1``.

        ``records_transmitted`` is the count the pipeline actually sent, not a
        plan. A local or blocked run reports zero and no provider, which is
        what the suite's reader requires of one.
        """
        if records_transmitted < 0:
            raise EgressError("records_transmitted cannot be negative.")
        if not self.content_leaves_machine and records_transmitted:
            raise EgressError("A run that sent nothing cannot report transmitted records.")
        used = self.content_leaves_machine and records_transmitted > 0
        document: dict[str, Any] = {
            "remote_provider_used": used,
            "records_transmitted": records_transmitted if used else 0,
            "raw_source_text_transmitted": used and not self.redaction_applied,
            "hosting_mode": self.hosting_mode,
            "hosted_execution_confirmed": self.hosted_execution_confirmed,
            "dataset_privacy_classification": self.dataset_privacy_classification,
            "redaction_applied": self.redaction_applied,
            "telemetry_enabled": self.telemetry_enabled,
        }
        if used:
            document["provider"] = self.provider
        if self.warnings:
            document["warnings"] = list(self.warnings)
        return document


def provider_data_preflight(
    *,
    provider: str | None = None,
    hosting_mode: str = LOCAL,
    dataset_privacy_classification: str = DEFAULT_CLASSIFICATION,
    allow_hosted: bool = False,
    telemetry_enabled: bool = False,
    redaction_applied: bool = False,
) -> EgressPreflight:
    """Decide whether this run may send source text anywhere, and say why.

    Every default is the safe one: local, unknown classification, not allowed,
    no telemetry, no redaction claimed. A caller who passes nothing gets a run
    that cannot send anything, which is the only default that cannot surprise
    anyone.
    """
    if hosting_mode not in HOSTING_MODES:
        raise EgressError(f"hosting_mode must be one of {sorted(HOSTING_MODES)}, not {hosting_mode!r}.")
    if dataset_privacy_classification not in PRIVACY_CLASSIFICATIONS:
        raise EgressError(
            f"dataset_privacy_classification must be one of {sorted(PRIVACY_CLASSIFICATIONS)}, not {dataset_privacy_classification!r}."
        )
    is_hosted = hosting_mode == HOSTED
    if is_hosted and (not provider or len(provider) > MAX_PROVIDER_CHARS):
        raise EgressError(f"A hosted run must name its provider, in 1 to {MAX_PROVIDER_CHARS} characters.")
    if not is_hosted and provider is not None:
        # A provider named on a local run is either a mistake or a claim the
        # record would carry without anything behind it.
        raise EgressError("A local run must not name a provider; nothing is sent to one.")

    warnings: list[str] = []
    if is_hosted and not allow_hosted:
        warnings.append(BLOCKED_WARNING)
    if is_hosted and dataset_privacy_classification not in OPEN_CLASSIFICATIONS and not redaction_applied:
        warnings.append(UNREDACTED_WARNING)
    if telemetry_enabled:
        warnings.append(TELEMETRY_WARNING)

    return EgressPreflight(
        provider=provider,
        hosting_mode=hosting_mode,
        dataset_privacy_classification=dataset_privacy_classification,
        hosted_execution_confirmed=allow_hosted,
        redaction_applied=redaction_applied,
        telemetry_enabled=telemetry_enabled,
        allowed=not is_hosted or allow_hosted,
        content_leaves_machine=is_hosted and allow_hosted,
        warnings=tuple(warnings[:MAX_WARNINGS]),
    )


def local_egress() -> EgressPreflight:
    """The default: nothing leaves the machine, and no socket is opened."""
    return provider_data_preflight()
