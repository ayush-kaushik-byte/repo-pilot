"""Tests for RepoPilot output validation."""

import pytest

from repopilot.guardrails.output_validation import (
    MAX_ANSWER_LENGTH,
    validate_output,
)


def test_valid_answer_returns_no_errors():
    assert validate_output("The request flow is implemented in views.py.") == []


@pytest.mark.parametrize("answer", ["", "   ", "\n\t"])
def test_empty_or_whitespace_answer_is_rejected(answer):
    assert "Answer must not be empty." in validate_output(answer)


@pytest.mark.parametrize("answer", [None, 123, [], {}])
def test_non_string_answer_is_rejected(answer):
    assert validate_output(answer) == ["Answer must be a string."]


def test_answer_exceeding_maximum_length_is_rejected():
    errors = validate_output("a" * (MAX_ANSWER_LENGTH + 1))

    assert any("must not exceed" in error for error in errors)


def test_answer_at_maximum_length_is_allowed():
    assert validate_output("a" * MAX_ANSWER_LENGTH) == []
