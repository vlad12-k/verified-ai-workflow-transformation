"""Deterministic semantic-drift risk checks for grounded RAG."""

import re
from dataclasses import dataclass
from enum import StrEnum


class SemanticDriftRisk(StrEnum):
    """Bounded classes of policy-semantic drift."""

    OBLIGATION_WEAKENED = "OBLIGATION_WEAKENED"
    EXCEPTION_QUALIFIER_DROPPED = "EXCEPTION_QUALIFIER_DROPPED"
    TEMPORAL_QUALIFIER_DROPPED = "TEMPORAL_QUALIFIER_DROPPED"
    SCOPE_QUALIFIER_DROPPED = "SCOPE_QUALIFIER_DROPPED"
    CONDITIONAL_MODALITY_DROPPED = "CONDITIONAL_MODALITY_DROPPED"
    UNIVERSAL_SCOPE_DROPPED = "UNIVERSAL_SCOPE_DROPPED"
    CAUSAL_RATIONALE_ADDED = "CAUSAL_RATIONALE_ADDED"


@dataclass(frozen=True)
class SemanticDriftIssue:
    """One conservative semantic-risk finding."""

    risk: SemanticDriftRisk
    marker: str
    detail: str


@dataclass(frozen=True)
class SemanticDriftAssessment:
    """Deterministic semantic-drift assessment."""

    passed: bool
    issues: tuple[SemanticDriftIssue, ...]


_HARD_OBLIGATION_WORD = re.compile(
    r"\b(?:must|required)\b",
    flags=re.IGNORECASE,
)

_REQUIRE_WORD = re.compile(
    r"\brequires?\b",
    flags=re.IGNORECASE,
)

_CONDITIONAL_REQUIRE = re.compile(
    r"\b(?:may|can|could|might)\s+requires?\b",
    flags=re.IGNORECASE,
)

_SENTENCE_INITIAL_ANY = re.compile(
    r"(?:^|[.!?]\s+)any\s+",
    flags=re.IGNORECASE,
)

_UNIVERSAL_MARKER = re.compile(
    r"\b(?:any|every|all|each)\b",
    flags=re.IGNORECASE,
)

_EXCEPTION_EQUIVALENTS = (
    "unless",
    "except when",
    "except if",
)

_BEFORE_EQUIVALENTS = (
    "before",
    "prior to",
    "only after",
    "not until",
)

_AFTER_EQUIVALENTS = (
    "after",
    "following",
    "once",
)

_NON_ZERO_EQUIVALENTS = (
    "non-zero",
    "nonzero",
    "other than zero",
    "not zero",
    "different from zero",
)

_LARGE_EQUIVALENTS = (
    "large",
    "significant",
    "substantial",
    "material",
)

_CONDITIONAL_EQUIVALENTS = (
    "may",
    "can",
    "could",
    "might",
)

_CAUSAL_MARKERS = (
    "because",
    "due to",
    "in order to",
    "so that",
    "therefore",
    "to ensure",
)


def assess_semantic_drift(
    *,
    context_text: str,
    answer_text: str,
) -> SemanticDriftAssessment:
    """Flag bounded indicators of policy-semantic drift."""
    if not context_text.strip():
        raise ValueError(
            "Semantic-drift context must not be empty."
        )

    if not answer_text.strip():
        raise ValueError(
            "Semantic-drift answer must not be empty."
        )

    context = _normalize(context_text)
    answer = _normalize(answer_text)

    issues: list[SemanticDriftIssue] = []

    if (
        _contains_hard_obligation(context)
        and not _preserves_hard_obligation(
            context=context,
            answer=answer,
        )
    ):
        issues.append(
            SemanticDriftIssue(
                risk=SemanticDriftRisk.OBLIGATION_WEAKENED,
                marker="hard-obligation",
                detail=(
                    "The evidence contains a hard obligation "
                    "that is not preserved by the answer."
                ),
            )
        )

    if (
        "unless" in context
        and not _contains_any(
            answer,
            _EXCEPTION_EQUIVALENTS,
        )
    ):
        issues.append(
            SemanticDriftIssue(
                risk=(
                    SemanticDriftRisk.EXCEPTION_QUALIFIER_DROPPED
                ),
                marker="unless",
                detail=(
                    "A material exception qualifier is not "
                    "preserved by the answer."
                ),
            )
        )

    if (
        "before" in context
        and not _contains_any(
            answer,
            _BEFORE_EQUIVALENTS,
        )
    ):
        issues.append(
            SemanticDriftIssue(
                risk=(
                    SemanticDriftRisk.TEMPORAL_QUALIFIER_DROPPED
                ),
                marker="before",
                detail=(
                    "A before-ordering constraint is not "
                    "preserved by the answer."
                ),
            )
        )

    if (
        "after" in context
        and not _contains_any(
            answer,
            _AFTER_EQUIVALENTS,
        )
    ):
        issues.append(
            SemanticDriftIssue(
                risk=(
                    SemanticDriftRisk.TEMPORAL_QUALIFIER_DROPPED
                ),
                marker="after",
                detail=(
                    "An after-ordering constraint is not "
                    "preserved by the answer."
                ),
            )
        )

    if (
        "non-zero" in context
        and not _contains_any(
            answer,
            _NON_ZERO_EQUIVALENTS,
        )
    ):
        issues.append(
            SemanticDriftIssue(
                risk=SemanticDriftRisk.SCOPE_QUALIFIER_DROPPED,
                marker="non-zero",
                detail=(
                    "The non-zero scope constraint is not "
                    "preserved by the answer."
                ),
            )
        )

    if (
        "large" in context
        and not _contains_any(
            answer,
            _LARGE_EQUIVALENTS,
        )
    ):
        issues.append(
            SemanticDriftIssue(
                risk=SemanticDriftRisk.SCOPE_QUALIFIER_DROPPED,
                marker="large",
                detail=(
                    "The large-difference scope qualifier is "
                    "not preserved by the answer."
                ),
            )
        )

    if (
        _contains_conditional_modality(context)
        and not _contains_any(
            answer,
            _CONDITIONAL_EQUIVALENTS,
        )
    ):
        issues.append(
            SemanticDriftIssue(
                risk=(
                    SemanticDriftRisk.CONDITIONAL_MODALITY_DROPPED
                ),
                marker="conditional-modality",
                detail=(
                    "A conditional policy modality is not "
                    "preserved by the answer."
                ),
            )
        )

    if (
        _SENTENCE_INITIAL_ANY.search(context)
        and not _UNIVERSAL_MARKER.search(answer)
    ):
        issues.append(
            SemanticDriftIssue(
                risk=SemanticDriftRisk.UNIVERSAL_SCOPE_DROPPED,
                marker="universal-any",
                detail=(
                    "A sentence-level universal scope is not "
                    "preserved by the answer."
                ),
            )
        )

    if (
        _contains_any(
            answer,
            _CAUSAL_MARKERS,
        )
        and not _contains_any(
            context,
            _CAUSAL_MARKERS,
        )
    ):
        issues.append(
            SemanticDriftIssue(
                risk=SemanticDriftRisk.CAUSAL_RATIONALE_ADDED,
                marker="causal-rationale",
                detail=(
                    "The answer introduces causal rationale "
                    "that is absent from the supplied evidence."
                ),
            )
        )

    return SemanticDriftAssessment(
        passed=not issues,
        issues=tuple(issues),
    )


def _contains_hard_obligation(
    text: str,
) -> bool:
    """Detect hard obligations without treating may-require as hard."""
    without_conditionals = _CONDITIONAL_REQUIRE.sub(
        "",
        text,
    )

    return bool(
        _HARD_OBLIGATION_WORD.search(
            without_conditionals
        )
        or _REQUIRE_WORD.search(
            without_conditionals
        )
    )


def _preserves_hard_obligation(
    *,
    context: str,
    answer: str,
) -> bool:
    """Recognize bounded hard-obligation paraphrases."""
    if _contains_hard_obligation(answer):
        return True

    if (
        "before" in context
        and "only after" in answer
    ):
        return True

    return (
        "before" in context
        and "not until" in answer
    )


def _contains_conditional_modality(
    text: str,
) -> bool:
    """Return whether evidence contains a conditional modal."""
    return _contains_any(
        text,
        _CONDITIONAL_EQUIVALENTS,
    )


def _contains_any(
    text: str,
    markers: tuple[str, ...],
) -> bool:
    """Return whether normalized text contains any marker."""
    return any(
        marker in text
        for marker in markers
    )


def _normalize(
    text: str,
) -> str:
    """Normalize text for conservative lexical comparison."""
    normalized = (
        text.casefold()
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )

    return re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()
