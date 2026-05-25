"""Robustness tests for the JSON parser used to read LLM responses.

The previous bare ``except ValueError`` swallowed parse failures silently;
``parse_json_object`` now raises a typed ``LLMParseError`` (still a
ValueError subclass for backward-compat) so we exercise both fenced and
malformed inputs.
"""

from __future__ import annotations

import pytest

from TRACE.src.scoring import parse_json_object
from TRACE.src.types import LLMParseError


def test_parses_plain_json_object():
    assert parse_json_object('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_strips_markdown_fences():
    raw = "```json\n{\"a\": 1}\n```"
    assert parse_json_object(raw) == {"a": 1}


def test_strips_bare_fences():
    raw = "```\n{\"a\": 1}\n```"
    assert parse_json_object(raw) == {"a": 1}


def test_handles_extra_whitespace():
    raw = "   \n  {\"a\": 1}   "
    assert parse_json_object(raw) == {"a": 1}


def test_raises_typed_error_on_malformed():
    with pytest.raises(LLMParseError):
        parse_json_object("{not json}")


def test_typed_error_is_value_error_for_legacy_callers():
    # Old call sites used ``except ValueError`` — they must keep working.
    with pytest.raises(ValueError):
        parse_json_object("nope")


def test_empty_string_raises():
    with pytest.raises(LLMParseError):
        parse_json_object("")
