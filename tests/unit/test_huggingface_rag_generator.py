"""Tests for the local Hugging Face RAG generator."""

from typing import Any

import pytest
import torch

from vait.rag.huggingface import HuggingFaceCausalGenerator
from vait.rag.models import (
    RAGContextDocument,
    RAGGenerationRequest,
)


class FakeTokenizer:
    """Minimal tokenizer double for deterministic generator tests."""

    eos_token_id = 99

    def __init__(
        self,
        *,
        decoded_answer: str = "Hold the invoice for investigation.",
    ) -> None:
        """Store deterministic decoded output."""
        self.decoded_answer = decoded_answer
        self.messages: list[dict[str, str]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Capture messages and return a deterministic prompt."""
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "encoded prompt"

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        """Return a small deterministic encoded prompt."""
        assert prompt == "encoded prompt"
        assert return_tensors == "pt"

        return {
            "input_ids": torch.tensor(
                [[1, 2, 3]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1]],
                dtype=torch.long,
            ),
        }

    def decode(
        self,
        tokens: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Return configured generated text."""
        assert tokens.numel() == 2
        assert skip_special_tokens is True
        return self.decoded_answer


class FakeModel:
    """Minimal causal-model double."""

    def __init__(self) -> None:
        """Track deterministic inference configuration."""
        self.device = "unconfigured"
        self.eval_called = False
        self.generate_kwargs: dict[str, Any] | None = None

    def to(
        self,
        device: str,
    ) -> "FakeModel":
        """Record target inference device."""
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        """Record evaluation mode."""
        self.eval_called = True
        return self

    def generate(
        self,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Append two deterministic generated token IDs."""
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]

        generated = torch.tensor(
            [[7, 8]],
            dtype=torch.long,
            device=input_ids.device,
        )

        return torch.cat(
            (
                input_ids,
                generated,
            ),
            dim=1,
        )


def build_request(
    *,
    score: float = 0.8,
) -> RAGGenerationRequest:
    """Build one grounded generation request."""
    return RAGGenerationRequest(
        query="What should happen to a duplicate invoice?",
        context_documents=(
            RAGContextDocument(
                document_id="duplicate-policy",
                text=(
                    "Duplicate invoices must be held "
                    "for investigation."
                ),
                score=score,
                rank=1,
            ),
        ),
    )


def patch_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer: FakeTokenizer,
    model: FakeModel,
) -> None:
    """Replace Hugging Face loading with deterministic doubles."""
    monkeypatch.setattr(
        "vait.rag.huggingface.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "vait.rag.huggingface.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )


def test_generator_loads_pinned_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned model metadata should be exposed reproducibly."""
    tokenizer = FakeTokenizer()
    model = FakeModel()
    patch_model_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    generator = HuggingFaceCausalGenerator(
        model_id="example/model",
        revision="abc123",
        device="cpu",
        minimum_score=0.4,
        max_new_tokens=64,
        local_files_only=True,
    )

    assert generator.model_id == "example/model"
    assert generator.revision == "abc123"
    assert generator.device == "cpu"
    assert generator.minimum_score == 0.4
    assert model.device == "cpu"
    assert model.eval_called is True
    assert "example/model@abc123" in generator.implementation_id


def test_generator_returns_generated_grounded_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strong evidence should invoke deterministic local generation."""
    tokenizer = FakeTokenizer()
    model = FakeModel()
    patch_model_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    generator = HuggingFaceCausalGenerator(
        model_id="example/model",
        revision="abc123",
        minimum_score=0.4,
        max_new_tokens=64,
    )

    generation = generator.generate(
        build_request()
    )

    assert generation.answer == (
        "Hold the invoice for investigation."
    )
    assert generation.cited_document_ids == (
        "duplicate-policy",
    )
    assert generation.abstained is False

    assert model.generate_kwargs is not None
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 64
    assert model.generate_kwargs["pad_token_id"] == 99

    assert tokenizer.messages is not None
    assert "duplicate-policy" in tokenizer.messages[1]["content"]


def test_generator_abstains_before_generation_for_weak_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieval evidence below threshold should not invoke the SLM."""
    tokenizer = FakeTokenizer()
    model = FakeModel()
    patch_model_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    generator = HuggingFaceCausalGenerator(
        model_id="example/model",
        revision="abc123",
        minimum_score=0.4,
    )

    generation = generator.generate(
        build_request(
            score=0.3,
        )
    )

    assert generation.abstained is True
    assert generation.cited_document_ids == ()
    assert model.generate_kwargs is None


def test_generator_supports_model_requested_abstention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit evidence sentinel should become an abstention."""
    tokenizer = FakeTokenizer(
        decoded_answer="INSUFFICIENT_EVIDENCE",
    )
    model = FakeModel()
    patch_model_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    generator = HuggingFaceCausalGenerator(
        model_id="example/model",
        revision="abc123",
    )

    generation = generator.generate(
        build_request()
    )

    assert generation.answer == "Insufficient evidence."
    assert generation.cited_document_ids == ()
    assert generation.abstained is True


def test_generator_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid configuration should fail before model loading."""
    tokenizer = FakeTokenizer()
    model = FakeModel()
    patch_model_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    with pytest.raises(
        ValueError,
        match="max_new_tokens",
    ):
        HuggingFaceCausalGenerator(
            model_id="example/model",
            revision="abc123",
            max_new_tokens=0,
        )


def test_generator_exposes_versioned_prompt_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt contract must be part of reproducible model identity."""
    from vait.rag.prompt_contracts import RAGPromptContract

    tokenizer = FakeTokenizer()
    model = FakeModel()
    patch_model_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    generator = HuggingFaceCausalGenerator(
        model_id="example/model",
        revision="abc123",
        prompt_contract=RAGPromptContract.STRICT_V2,
    )

    assert generator.prompt_contract is RAGPromptContract.STRICT_V2
    assert (
        "prompt-contract=strict-v2"
        in generator.implementation_id
    )


def test_generator_exposes_exact_usage_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured generation should expose exact prompt and output token counts."""
    tokenizer = FakeTokenizer()
    model = FakeModel()

    patch_model_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    generator = HuggingFaceCausalGenerator(
        model_id="example/model",
        revision="abc123",
        minimum_score=0.4,
        max_new_tokens=64,
    )

    evidence = generator.generate_with_evidence(
        build_request()
    )

    assert evidence.generation.answer == (
        "Hold the invoice for investigation."
    )
    assert evidence.generation.abstained is False

    assert evidence.input_tokens == 3
    assert evidence.output_tokens == 2
    assert evidence.model_generation_latency_ms >= 0.0


def test_generator_usage_evidence_preserves_pre_generation_abstention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weak evidence should report zero model usage when generation is skipped."""
    tokenizer = FakeTokenizer()
    model = FakeModel()

    patch_model_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    generator = HuggingFaceCausalGenerator(
        model_id="example/model",
        revision="abc123",
        minimum_score=0.4,
    )

    evidence = generator.generate_with_evidence(
        build_request(
            score=0.3,
        )
    )

    assert evidence.generation.abstained is True
    assert evidence.input_tokens == 0
    assert evidence.output_tokens == 0
    assert evidence.model_generation_latency_ms == 0.0
    assert model.generate_kwargs is None
