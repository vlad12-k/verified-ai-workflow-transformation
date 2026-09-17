"""Tests for AP policy specialist instruction-tuning data."""

from vait.rag.specialist_training import (
    build_ap_policy_specialist_corpus,
)

BENCHMARK_QUERIES = {
    "What should happen if the same supplier invoice appears twice?",
    "What must happen before paying a supplier after its bank account changes?",
    "How should an invoice with no purchase order reference be handled?",
    "What should happen when the invoice amount differs from the purchase order?",
    "What if the vendor is not in the approved supplier master?",
    "What should happen when a supplier tax identifier cannot be validated?",
    "How should a zero-value or negative invoice be handled?",
    "Who should approve an invoice that exceeds the normal delegated limit?",
    "How many days of annual leave can an employee carry into next year?",
    "How long is the probation period for a newly hired employee?",
    "Which hotels may employees book for international business travel?",
    "Who approves the quarterly social media advertising budget?",
    "How often should physical warehouse inventory be counted?",
    "What is the replacement cycle for employee laptops and monitors?",
}


def test_specialist_corpus_has_train_and_dev_splits() -> None:
    """The specialist corpus must contain both train and dev data."""
    corpus = build_ap_policy_specialist_corpus()

    assert len(corpus.train_examples) == 11
    assert len(corpus.dev_examples) == 3


def test_specialist_corpus_has_unique_ids() -> None:
    """Every specialist example must have one stable identity."""
    corpus = build_ap_policy_specialist_corpus()

    example_ids = [
        example.example_id
        for example in corpus.examples
    ]

    assert len(example_ids) == len(set(example_ids))


def test_specialist_corpus_excludes_exact_benchmark_queries() -> None:
    """Training and dev prompts must not copy benchmark questions."""
    corpus = build_ap_policy_specialist_corpus()

    corpus_queries = {
        example.question
        for example in corpus.examples
    }

    assert corpus_queries.isdisjoint(
        BENCHMARK_QUERIES
    )


def test_specialist_corpus_contains_abstention_examples() -> None:
    """Training must include explicit unsupported-query behaviour."""
    corpus = build_ap_policy_specialist_corpus()

    train_abstentions = [
        example
        for example in corpus.train_examples
        if example.answer == "INSUFFICIENT_EVIDENCE"
    ]

    dev_abstentions = [
        example
        for example in corpus.dev_examples
        if example.answer == "INSUFFICIENT_EVIDENCE"
    ]

    assert len(train_abstentions) >= 2
    assert len(dev_abstentions) >= 1


def test_specialist_corpus_fingerprint_is_reproducible() -> None:
    """Repeated construction must preserve corpus identity."""
    first = build_ap_policy_specialist_corpus()
    second = build_ap_policy_specialist_corpus()

    assert first.sha256 == second.sha256


def test_specialist_corpus_answers_are_non_empty() -> None:
    """Every supervised target must contain an answer."""
    corpus = build_ap_policy_specialist_corpus()

    assert all(
        example.answer.strip()
        for example in corpus.examples
    )
