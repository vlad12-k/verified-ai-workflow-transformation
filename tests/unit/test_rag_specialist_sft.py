"""Tests for specialist supervised fine-tuning formatting."""

import pytest

from vait.rag.specialist_sft import (
    IGNORE_INDEX,
    build_specialist_messages,
    format_specialist_sft_sample,
)
from vait.rag.specialist_training import (
    SpecialistTrainingExample,
)


class FakeChatTokenizer:
    """Minimal deterministic tokenizer for SFT unit tests."""

    eos_token_id = 0

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Render a deterministic role-tagged conversation."""
        assert tokenize is False

        rendered = "".join(
            f"<{message['role']}>"
            f"{message['content']}"
            f"</{message['role']}>"
            for message in conversation
        )

        if add_generation_prompt:
            rendered += "<assistant>"

        return rendered

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        """Map every character to one deterministic integer token."""
        assert add_special_tokens is False

        return {
            "input_ids": [
                ord(character)
                for character in text
            ]
        }


def build_example() -> SpecialistTrainingExample:
    """Return one compact grounded specialist example."""
    return SpecialistTrainingExample(
        example_id="train-test",
        split="train",
        policy_context=(
            "Invoices above the threshold require approval."
        ),
        question="What is required?",
        answer="Approval is required.",
    )


def test_messages_include_grounded_context_and_question() -> None:
    """Specialist messages should preserve evidence and query."""
    example = build_example()

    messages = build_specialist_messages(
        example,
        include_answer=False,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert example.policy_context in messages[1]["content"]
    assert example.question in messages[1]["content"]


def test_answer_is_added_only_when_requested() -> None:
    """Training conversations should add one assistant target."""
    example = build_example()

    messages = build_specialist_messages(
        example,
        include_answer=True,
    )

    assert len(messages) == 3
    assert messages[-1] == {
        "role": "assistant",
        "content": example.answer,
    }


def test_sft_sample_masks_prompt_tokens() -> None:
    """Loss labels must ignore system and user prompt tokens."""
    sample = format_specialist_sft_sample(
        build_example(),
        tokenizer=FakeChatTokenizer(),
    )

    assert all(
        label == IGNORE_INDEX
        for label in sample.labels[
            : sample.prompt_token_count
        ]
    )


def test_sft_sample_supervises_assistant_tokens() -> None:
    """Assistant response tokens must remain supervised."""
    sample = format_specialist_sft_sample(
        build_example(),
        tokenizer=FakeChatTokenizer(),
    )

    supervised_labels = sample.labels[
        sample.prompt_token_count :
    ]

    assert supervised_labels
    assert all(
        label != IGNORE_INDEX
        for label in supervised_labels
    )
    assert sample.supervised_token_count == len(
        supervised_labels
    )


def test_sft_sample_shapes_match() -> None:
    """Input, attention, and label tensors must align."""
    sample = format_specialist_sft_sample(
        build_example(),
        tokenizer=FakeChatTokenizer(),
    )

    assert len(sample.input_ids) == len(
        sample.attention_mask
    )
    assert len(sample.input_ids) == len(
        sample.labels
    )
    assert sample.token_count == len(
        sample.input_ids
    )
    assert set(sample.attention_mask) == {1}


def test_sft_sample_rejects_non_positive_max_length() -> None:
    """Invalid sequence limits must fail explicitly."""
    with pytest.raises(
        ValueError,
        match="max_length must be greater than zero",
    ):
        format_specialist_sft_sample(
            build_example(),
            tokenizer=FakeChatTokenizer(),
            max_length=0,
        )


def test_sft_sample_rejects_truncation_requirement() -> None:
    """Examples longer than the declared bound must fail closed."""
    with pytest.raises(
        ValueError,
        match="exceeds max_length",
    ):
        format_specialist_sft_sample(
            build_example(),
            tokenizer=FakeChatTokenizer(),
            max_length=5,
        )
