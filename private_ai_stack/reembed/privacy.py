"""Validated consent and evidence for re-embedding executor boundaries.

The built-in registry and an injected executor are different trust boundaries.
Registry-owned embedders are deterministic package code. An injected executor
is arbitrary operator code and may use providers, proxies, redirects, telemetry,
or other network paths that this package cannot observe. Its network behaviour
is therefore attested, never described as isolated or scanner-verified.

Consent is checked before the source evidence document is opened. The pipeline
derives transmission fields from payloads it actually submits; operator flags
cannot claim that untransformed text was redacted.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlsplit

LOCAL: Final = "local"
HOSTED: Final = "hosted"
HOSTING_MODES: Final = frozenset({LOCAL, HOSTED})

LOCAL_BUILTIN: Final = "local_builtin"
EXTERNAL_UNVERIFIED: Final = "external_unverified"
EXECUTOR_KINDS: Final = frozenset({LOCAL_BUILTIN, EXTERNAL_UNVERIFIED})

PRIVACY_CLASSIFICATIONS: Final = frozenset({"public", "internal", "confidential", "restricted", "unknown"})
DEFAULT_CLASSIFICATION: Final = "unknown"
HOSTED_BLOCKED_CLASSIFICATIONS: Final = frozenset({"unknown", "confidential", "restricted"})

BLOCKED_WARNING: Final = "external execution is blocked until explicitly enabled"
HOSTED_BLOCKED_WARNING: Final = "hosted execution is blocked until explicitly enabled"
CLASSIFICATION_BLOCKED_WARNING: Final = "hosted execution is blocked for this data classification"
UNREDACTED_WARNING: Final = "private unredacted dataset content would be transmitted to a hosted provider"
TELEMETRY_WARNING: Final = "telemetry is enabled; review exported fields before execution"
EXTERNAL_ATTESTED_WARNING: Final = "external executor network behaviour is operator-attested and unverified"
MAX_WARNINGS: Final = 16
MAX_PROVIDER_CHARS: Final = 128
MAX_ENDPOINT_CHARS: Final = 2048


class EgressError(ValueError):
    """Raised when an executor preflight is invalid or does not grant consent."""


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise EgressError(f"{name} must be a boolean.")
    return bool(value)


def _endpoint_origin(endpoint: str, hosting_mode: str) -> str:
    """Return a non-secret origin after validating the declared endpoint.

    This validates only the operator's declaration. An external executor can
    ignore it, so no caller may use this result as proof of its network route.
    """
    if not endpoint or len(endpoint) > MAX_ENDPOINT_CHARS or any(ord(char) < 32 for char in endpoint):
        raise EgressError("The external endpoint is invalid.")
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        raise EgressError("The external endpoint is invalid.") from None
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise EgressError("The external endpoint must not contain credentials, a query, or a fragment.")
    if not parsed.hostname or parsed.scheme not in {"http", "https"}:
        raise EgressError("The external endpoint must be an absolute HTTP(S) URL.")

    hostname = parsed.hostname.lower().rstrip(".")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None

    if hosting_mode == LOCAL:
        if hostname != "localhost" and (address is None or not address.is_loopback):
            raise EgressError("A local external endpoint must use a loopback address.")
    else:
        if parsed.scheme != "https":
            raise EgressError("A hosted external endpoint must use HTTPS.")
        if hostname == "localhost" or (address is not None and not address.is_global):
            raise EgressError("A hosted external endpoint must not use a local or private address.")

    default_port = 443 if parsed.scheme == "https" else 80
    authority = hostname if port in (None, default_port) else f"{hostname}:{port}"
    return f"{parsed.scheme}://{authority}"


def _warnings(
    *,
    executor_kind: str,
    hosting_mode: str,
    classification: str,
    external_confirmed: bool,
    hosted_confirmed: bool,
    telemetry_enabled: bool,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if executor_kind == EXTERNAL_UNVERIFIED:
        warnings.append(EXTERNAL_ATTESTED_WARNING)
        if not external_confirmed:
            warnings.append(BLOCKED_WARNING)
        if hosting_mode == HOSTED and not hosted_confirmed:
            warnings.append(HOSTED_BLOCKED_WARNING)
        if hosting_mode == HOSTED and classification in HOSTED_BLOCKED_CLASSIFICATIONS:
            warnings.append(CLASSIFICATION_BLOCKED_WARNING)
        if hosting_mode == HOSTED and classification == "internal":
            warnings.append(UNREDACTED_WARNING)
        if telemetry_enabled:
            warnings.append(TELEMETRY_WARNING)
    return tuple(warnings[:MAX_WARNINGS])


@dataclass(frozen=True)
class EgressPreflight:
    """Validated permission and attested network context for one executor."""

    executor_kind: str
    provider: str | None
    endpoint_origin: str | None
    hosting_mode: str
    dataset_privacy_classification: str
    external_execution_confirmed: bool
    hosted_execution_confirmed: bool
    redaction_applied: bool
    telemetry_enabled: bool
    redirects_allowed: bool
    proxy_url: str | None
    allowed: bool
    content_leaves_machine: bool
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.executor_kind not in EXECUTOR_KINDS:
            raise EgressError(f"executor_kind must be one of {sorted(EXECUTOR_KINDS)}.")
        if self.hosting_mode not in HOSTING_MODES:
            raise EgressError(f"hosting_mode must be one of {sorted(HOSTING_MODES)}.")
        if self.dataset_privacy_classification not in PRIVACY_CLASSIFICATIONS:
            raise EgressError(f"dataset_privacy_classification must be one of {sorted(PRIVACY_CLASSIFICATIONS)}.")
        for name in (
            "external_execution_confirmed",
            "hosted_execution_confirmed",
            "redaction_applied",
            "telemetry_enabled",
            "redirects_allowed",
            "allowed",
            "content_leaves_machine",
        ):
            _exact_bool(getattr(self, name), name)
        if not isinstance(self.warnings, tuple) or not all(isinstance(item, str) and item for item in self.warnings):
            raise EgressError("warnings must be a tuple of non-empty strings.")
        if len(self.warnings) > MAX_WARNINGS:
            raise EgressError(f"warnings cannot contain more than {MAX_WARNINGS} entries.")
        if self.redaction_applied:
            raise EgressError("Redaction is not implemented; untransformed source text cannot be declared redacted.")
        if self.proxy_url is not None:
            raise EgressError("Proxy routing is unsupported for external executors.")
        if self.redirects_allowed:
            raise EgressError("Redirect following is unsupported for external executors.")

        external = self.executor_kind == EXTERNAL_UNVERIFIED
        hosted = self.hosting_mode == HOSTED
        if not external:
            if hosted or self.provider is not None or self.endpoint_origin is not None:
                raise EgressError("A local_builtin executor cannot declare a provider, endpoint, or hosted mode.")
            if self.external_execution_confirmed or self.hosted_execution_confirmed or self.telemetry_enabled:
                raise EgressError("A local_builtin executor cannot claim external consent or provider telemetry.")
        else:
            if not self.provider or len(self.provider) > MAX_PROVIDER_CHARS:
                raise EgressError(f"An external executor must have an identifier of 1 to {MAX_PROVIDER_CHARS} characters.")
            if self.external_execution_confirmed and self.endpoint_origin is None:
                raise EgressError("An enabled external executor must declare its endpoint.")
            if self.endpoint_origin is not None:
                normalized = _endpoint_origin(self.endpoint_origin, self.hosting_mode)
                if normalized != self.endpoint_origin:
                    raise EgressError("endpoint_origin must contain only the normalized endpoint origin.")
            if not hosted and self.hosted_execution_confirmed:
                raise EgressError("A local external executor cannot claim hosted execution consent.")

        classification_allowed = not hosted or self.dataset_privacy_classification not in HOSTED_BLOCKED_CLASSIFICATIONS
        expected_allowed = (not external) or (
            self.external_execution_confirmed and (not hosted or self.hosted_execution_confirmed) and classification_allowed
        )
        expected_leaves = external and hosted and expected_allowed
        expected_warnings = _warnings(
            executor_kind=self.executor_kind,
            hosting_mode=self.hosting_mode,
            classification=self.dataset_privacy_classification,
            external_confirmed=self.external_execution_confirmed,
            hosted_confirmed=self.hosted_execution_confirmed,
            telemetry_enabled=self.telemetry_enabled,
        )
        if self.allowed != expected_allowed or self.content_leaves_machine != expected_leaves:
            raise EgressError("Preflight permission fields contradict the declared executor and consent.")
        if self.warnings != expected_warnings:
            raise EgressError("Preflight warnings contradict the declared executor and consent.")

    @property
    def is_hosted(self) -> bool:
        return self.hosting_mode == HOSTED

    @property
    def is_external(self) -> bool:
        return self.executor_kind == EXTERNAL_UNVERIFIED

    @property
    def blocked(self) -> bool:
        return not self.allowed

    @property
    def execution_fingerprint(self) -> str:
        """Bind resumable work to the same trust boundary and declarations."""
        payload = {
            "executor_kind": self.executor_kind,
            "provider": self.provider,
            "endpoint_origin": self.endpoint_origin,
            "hosting_mode": self.hosting_mode,
            "dataset_privacy_classification": self.dataset_privacy_classification,
            "external_execution_confirmed": self.external_execution_confirmed,
            "hosted_execution_confirmed": self.hosted_execution_confirmed,
            "telemetry_enabled": self.telemetry_enabled,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_document(self, *, records_submitted: int) -> dict[str, Any]:
        """Build evidence from the number of distinct payloads submitted."""
        if not isinstance(records_submitted, int) or isinstance(records_submitted, bool) or records_submitted < 0:
            raise EgressError("records_submitted must be a non-negative integer.")
        if not self.is_external and records_submitted:
            raise EgressError("A local_builtin executor cannot report external payloads.")
        if self.blocked and records_submitted:
            raise EgressError("A blocked executor cannot report submitted payloads.")

        remotely_used = self.content_leaves_machine and records_submitted > 0
        document: dict[str, Any] = {
            "executor_kind": self.executor_kind,
            "execution_verification": "attested" if self.is_external else "registry_owned",
            "remote_provider_used": remotely_used,
            "records_transmitted": records_submitted if remotely_used else 0,
            "records_submitted_to_external_executor": records_submitted if self.is_external else 0,
            "raw_source_text_transmitted": remotely_used,
            "hosting_mode": self.hosting_mode,
            "external_execution_confirmed": self.external_execution_confirmed,
            "hosted_execution_confirmed": self.hosted_execution_confirmed,
            "dataset_privacy_classification": self.dataset_privacy_classification,
            "redaction_applied": False,
            "telemetry_enabled": self.telemetry_enabled,
        }
        if self.is_external:
            document["executor_id"] = self.provider
            document["endpoint_origin"] = self.endpoint_origin
            document["network_behavior_verification"] = "attested"
        if remotely_used:
            document["provider"] = self.provider
        if self.warnings:
            document["warnings"] = list(self.warnings)
        return document


def provider_data_preflight(
    *,
    executor_kind: str = LOCAL_BUILTIN,
    provider: str | None = None,
    endpoint: str | None = None,
    hosting_mode: str = LOCAL,
    dataset_privacy_classification: str = DEFAULT_CLASSIFICATION,
    allow_external: bool = False,
    allow_hosted: bool = False,
    telemetry_enabled: bool = False,
    redaction_applied: bool = False,
    proxy_url: str | None = None,
    allow_redirects: bool = False,
) -> EgressPreflight:
    """Validate declarations and derive permission without reading source data."""
    external_confirmed = _exact_bool(allow_external, "allow_external")
    hosted_confirmed = _exact_bool(allow_hosted, "allow_hosted")
    telemetry = _exact_bool(telemetry_enabled, "telemetry_enabled")
    redacted = _exact_bool(redaction_applied, "redaction_applied")
    redirects = _exact_bool(allow_redirects, "allow_redirects")
    if executor_kind not in EXECUTOR_KINDS:
        raise EgressError(f"executor_kind must be one of {sorted(EXECUTOR_KINDS)}.")
    if hosting_mode not in HOSTING_MODES:
        raise EgressError(f"hosting_mode must be one of {sorted(HOSTING_MODES)}.")
    if dataset_privacy_classification not in PRIVACY_CLASSIFICATIONS:
        raise EgressError(f"dataset_privacy_classification must be one of {sorted(PRIVACY_CLASSIFICATIONS)}.")
    if redacted:
        raise EgressError("Redaction is not implemented; untransformed source text cannot be declared redacted.")
    if proxy_url is not None:
        raise EgressError("Proxy routing is unsupported for external executors.")
    if redirects:
        raise EgressError("Redirect following is unsupported for external executors.")

    external = executor_kind == EXTERNAL_UNVERIFIED
    hosted = hosting_mode == HOSTED
    origin = None
    if external:
        if not provider or len(provider) > MAX_PROVIDER_CHARS:
            raise EgressError(f"An external executor must have an identifier of 1 to {MAX_PROVIDER_CHARS} characters.")
        if endpoint is not None:
            origin = _endpoint_origin(endpoint, hosting_mode)
        if external_confirmed and origin is None:
            raise EgressError("An enabled external executor must declare its endpoint.")
    elif provider is not None or endpoint is not None or hosted:
        raise EgressError("A local_builtin executor cannot declare a provider, endpoint, or hosted mode.")

    classification_allowed = not hosted or dataset_privacy_classification not in HOSTED_BLOCKED_CLASSIFICATIONS
    allowed = (not external) or (external_confirmed and (not hosted or hosted_confirmed) and classification_allowed)
    content_leaves = external and hosted and allowed
    warnings = _warnings(
        executor_kind=executor_kind,
        hosting_mode=hosting_mode,
        classification=dataset_privacy_classification,
        external_confirmed=external_confirmed,
        hosted_confirmed=hosted_confirmed,
        telemetry_enabled=telemetry,
    )
    return EgressPreflight(
        executor_kind=executor_kind,
        provider=provider,
        endpoint_origin=origin,
        hosting_mode=hosting_mode,
        dataset_privacy_classification=dataset_privacy_classification,
        external_execution_confirmed=external_confirmed,
        hosted_execution_confirmed=hosted_confirmed,
        redaction_applied=False,
        telemetry_enabled=telemetry,
        redirects_allowed=False,
        proxy_url=None,
        allowed=allowed,
        content_leaves_machine=content_leaves,
        warnings=warnings,
    )


def local_egress() -> EgressPreflight:
    """Return the compatible default for a registry-owned local embedder."""
    return provider_data_preflight()
