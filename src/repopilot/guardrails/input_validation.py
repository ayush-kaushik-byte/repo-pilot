"""Input validation guardrails for RepoPilot."""

import re


MAX_QUERY_LENGTH = 4_000
MAX_MODEL_NAME_LENGTH = 200

_OWNER_PATTERN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?"
)
_REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,100}")

_DANGEROUS_OFF_TOPIC_PATTERN = re.compile(
    r"\b(?:"
    r"how\s+(?:to|can\s+i|do\s+i)|"
    r"instructions?\s+(?:for|to)|"
    r"steps?\s+to|"
    r"guide\s+to|"
    r"teach\s+me\s+to|"
    r"tutorial\s+(?:for|on|to)"
    r")\b"
    r".{0,100}?"
    r"\b(?:make|build|construct|assemble|fabricate)\b"
    r".{0,50}?"
    r"\b(?:nuclear|atomic)\s+(?:bomb|weapon|warhead)s?\b",
    re.IGNORECASE,
)

def validate_query_scope(query: str) -> list[str]:
    """Reject clearly harmful requests outside repository analysis."""
    if _DANGEROUS_OFF_TOPIC_PATTERN.search(query):
        return [
            "Query is outside RepoPilot's scope. "
            "RepoPilot supports repository analysis and cannot "
            "assist with instructions for constructing nuclear weapons."
        ]

    return []

def validate_request(
    owner: object,
    repository: object,
    query: object,
    model_name: object = "llama3.2:3b",
) -> list[str]:
    """Return validation errors; an empty list means the request is valid."""
    errors: list[str] = []

    if not isinstance(owner, str) or not _OWNER_PATTERN.fullmatch(owner):
        errors.append(
            "Owner must be a valid GitHub owner identifier."
        )

    if (
        not isinstance(repository, str)
        or not _REPOSITORY_PATTERN.fullmatch(repository)
        or repository in {".", ".."}
        or repository.endswith(".")
    ):
        errors.append(
            "Repository must be a valid GitHub repository name."
        )

    if not isinstance(query, str) or not query.strip():
        errors.append("Query must be a non-empty string.")
    elif len(query) > MAX_QUERY_LENGTH:
        errors.append(
            f"Query must not exceed {MAX_QUERY_LENGTH} characters."
        )

    if (
        not isinstance(model_name, str)
        or not model_name.strip()
        or len(model_name) > MAX_MODEL_NAME_LENGTH
        or any(ord(char) < 32 for char in model_name)
    ):
        errors.append("Model name is missing or invalid.")
    
    if isinstance(query, str) and len(query) <= MAX_QUERY_LENGTH:
        errors.extend(validate_query_scope(query))

    return errors