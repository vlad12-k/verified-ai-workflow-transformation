"""Qwen-compatible supervised fine-tuning formatting for AP specialists."""

from dataclasses import dataclass
from typing import Any, Protocol

from vait.rag.specialist_training import (
    SpecialistTrainingExample,
)

IGNORE_INDEX = -100

SYSTEM_MESSAGE = (
    "Answer only from the supplied policy context. "
    "Preserve all material conditions, exceptions, qualifiers, and modality. "
    "Do not add unsupported facts or rationale. "
    "If the context does not support the answer, respond exactly "
    "INSUFFICIENT_EVIDENCE."
)


class ChatTokenizer(Protocol):
    """Typed tokenizer boundary used by specialist SFT formatting."""

    eos_token_id: int | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any:
        """Render one chat conversation."""
        ...

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> dict[str, Any]:
        """Tokenize text into model IDs."""
        ...


@dataclass(frozen=True)
class SpecialistSFTSample:
    """One tokenized causal-LM supervised training example."""

    example_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_token_count: int
    supervised_token_count: int

    @property
    def token_count(self) -> int:
        """Return total sequence length."""
        return len(self.input_ids)


def build_specialist_messages(
    example: SpecialistTrainingExample,
    *,
    include_answer: bool,
) -> list[dict[str, str]]:
    """Build one grounded specialist chat conversation."""
    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": (
                "Policy context:\n"
                f"{example.policy_context}\n\n"
                "Question:\n"
                f"{example.question}"
            ),
        },
    ]

    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": example.answer,
            }
        )

    return messages


def format_specialist_sft_sample(
    example: SpecialistTrainingExample,
    *,
    tokenizer: ChatTokenizer,
    max_length: int = 512,
) -> SpecialistSFTSample:
    """Format one example with assistant-only supervised labels."""
    if max_length <= 0:
        raise ValueError(
            "max_length must be greater than zero."
        )

    prompt_messages = build_specialist_messages(
        example,
        include_answer=False,
    )

    full_messages = build_specialist_messages(
        example,
        include_answer=True,
    )

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if not isinstance(prompt_text, str):
        raise TypeError(
            "Tokenizer chat template must return text."
        )

    if not isinstance(full_text, str):
        raise TypeError(
            "Tokenizer chat template must return text."
        )

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )

    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
    )

    prompt_ids = _extract_input_ids(
        prompt_tokens
    )
    input_ids = _extract_input_ids(
        full_tokens
    )

    if len(input_ids) > max_length:
        raise ValueError(
            "Specialist SFT example exceeds max_length: "
            f"{example.example_id} has {len(input_ids)} tokens."
        )

    if len(prompt_ids) >= len(input_ids):
        raise ValueError(
            "Assistant target must contribute at least one token."
        )

    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt tokenization is not a prefix of the "
            "full training conversation."
        )

    labels = (
        [IGNORE_INDEX] * len(prompt_ids)
        + input_ids[len(prompt_ids) :]
    )

    attention_mask = [1] * len(input_ids)

    supervised_token_count = sum(
        label != IGNORE_INDEX
        for label in labels
    )

    return SpecialistSFTSample(
        example_id=example.example_id,
        input_ids=tuple(input_ids),
        attention_mask=tuple(attention_mask),
        labels=tuple(labels),
        prompt_token_count=len(prompt_ids),
        supervised_token_count=supervised_token_count,
    )


def _extract_input_ids(
    encoded: dict[str, Any],
) -> list[int]:
    """Extract one flat token-ID sequence from tokenizer output."""
    input_ids = encoded.get("input_ids")

    if not isinstance(input_ids, list):
        raise TypeError(
            "Tokenizer input_ids must be a list."
        )

    if not all(
        isinstance(token_id, int)
        for token_id in input_ids
    ):
        raise TypeError(
            "Tokenizer input_ids must contain integers."
        )

    return input_ids
