
"""Output validation guardrails for RepoPilot."""

MAX_ANSWER_LENGTH = 20_000


def validate_output(answer: object) -> list[str]:
    """Return validation errors; an empty list means the output is valid."""
    errors: list[str] = []

    if not isinstance(answer, str):
        errors.append("Answer must be a string.")
        return errors

    if not answer.strip():
        errors.append("Answer must not be empty.")

    if len(answer) > MAX_ANSWER_LENGTH:
        errors.append(
            f"Answer must not exceed {MAX_ANSWER_LENGTH} characters."
        )

    return errors
