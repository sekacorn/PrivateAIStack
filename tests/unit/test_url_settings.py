"""Configured URLs carry no credentials, query string or fragment.

The HTTP client logs each request URL and the Ollama URL is handed on to
Forge, so a secret placed in either setting would reach logs. Such a URL is
refused when the settings load, and the refusal never repeats it.
"""

from __future__ import annotations

import traceback

import pytest
from pydantic import ValidationError

from private_ai_stack.config.settings import Settings

PLANTED = "url-setting-planted-4a7d-do-not-log"
BAD = {
    "credentials": (f"http://user:{PLANTED}@ollama:11434", "must not contain credentials"),
    "password only": (f"http://:{PLANTED}@ollama:11434", "must not contain credentials"),
    "query": (f"http://ollama:11434?key={PLANTED}", "must not contain a query string"),
    "empty query": ("http://ollama:11434?", "must not contain a query string"),
    "fragment": (f"http://ollama:11434#{PLANTED}", "must not contain a fragment"),
}


def _rendered(error: BaseException) -> str:
    return "\n".join([str(error), repr(error), "".join(traceback.format_exception(type(error), error, error.__traceback__))])


@pytest.mark.parametrize("setting", ["ollama_base_url", "otel_endpoint"])
@pytest.mark.parametrize("case", sorted(BAD))
def test_a_url_setting_with_a_secret_in_it_is_refused_without_echo(case: str, setting: str) -> None:
    url, message = BAD[case]

    with pytest.raises(ValidationError) as raised:
        Settings(**{setting: url})

    assert message in str(raised.value)
    assert PLANTED not in _rendered(raised.value)


def test_a_rejected_database_url_is_not_echoed_either() -> None:
    # A DSN carries its password by design; a rejected one must not be printed.
    with pytest.raises(ValidationError) as raised:
        Settings(database_url=f"mysql://user:{PLANTED}@db/app")

    assert PLANTED not in _rendered(raised.value)


@pytest.mark.parametrize("url", ["http://ollama:11434", "https://ollama.internal.example:443/", "http://127.0.0.1:11434"])
def test_a_clean_url_is_accepted_and_normalized(url: str) -> None:
    settings = Settings(ollama_base_url=url, otel_endpoint=url)

    assert settings.ollama_base_url == url.rstrip("/")
    assert settings.otel_endpoint == url.rstrip("/")


def test_the_otel_endpoint_stays_optional() -> None:
    assert Settings(otel_endpoint=None).otel_endpoint is None
