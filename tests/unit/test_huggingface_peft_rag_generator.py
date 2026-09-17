"""Tests for the local PEFT-adapted Hugging Face RAG generator."""

from pathlib import Path
from typing import Any

import pytest
import torch

from vait.rag.huggingface_peft import (
    HuggingFacePeftCausalGenerator,
)
from vait.rag.models import (
    RAGContextDocument,
    RAGGenerationRequest,
)
from vait.rag.prompt_contracts import (
    RAGPromptContract,
)


class FakeTokenizer:
    """Minimal tokenizer double for deterministic PEFT tests."""

    eos_token_id = 99

    def __init__(
        self,
        *,
        decoded_answer: str = "Hold the invoice.",
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
        """Return deterministic prompt tensors."""
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
    """Minimal PEFT causal-model double."""

    def __init__(self) -> None:
        """Track inference configuration."""
        self.device = "unconfigured"
        self.eval_called = False
        self.generate_kwargs: dict[str, Any] | None = None

    def to(
        self,
        device: str,
    ) -> "FakeModel":
        """Record target device."""
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


def patch_loading(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer: FakeTokenizer,
    model: FakeModel,
) -> None:
    """Replace Hugging Face and PEFT loading with doubles."""
    monkeypatch.setattr(
        "vait.rag.huggingface_peft.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    monkeypatch.setattr(
        (
            "vait.rag.huggingface_peft."
            "AutoModelForCausalLM.from_pretrained"
        ),
        lambda *args, **kwargs: object(),
    )

    monkeypatch.setattr(
        "vait.rag.huggingface_peft.PeftModel.from_pretrained",
        lambda *args, **kwargs: model,
    )


def build_generator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    decoded_answer: str = "Hold the invoice.",
    minimum_score: float = 0.4,
) -> tuple[
    HuggingFacePeftCausalGenerator,
    FakeTokenizer,
    FakeModel,
]:
    """Build one deterministic PEFT generator."""
    tokenizer = FakeTokenizer(
        decoded_answer=decoded_answer
    )
    model = FakeModel()

    patch_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()

    generator = HuggingFacePeftCausalGenerator(
        model_id="example/model",
        revision="abc123",
        adapter_path=adapter_path,
        device="cpu",
        minimum_score=minimum_score,
        max_new_tokens=64,
        local_files_only=True,
        prompt_contract=RAGPromptContract.STRICT_V2,
    )

    return generator, tokenizer, model


def test_peft_generator_loads_reproducible_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pinned base and adapter metadata should be exposed."""
    generator, _, model = build_generator(
        monkeypatch,
        tmp_path,
    )

    assert generator.model_id == "example/model"
    assert generator.revision == "abc123"
    assert generator.device == "cpu"
    assert generator.minimum_score == 0.4
    assert generator.prompt_contract is RAGPromptContract.STRICT_V2
    assert generator.adapter_path.name == "adapter"

    assert model.device == "cpu"
    assert model.eval_called is True

    assert "example/model@abc123" in generator.implementation_id
    assert "adapter=adapter" in generator.implementation_id
    assert "prompt-contract=strict-v2" in generator.implementation_id


def test_peft_generator_returns_grounded_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Strong evidence should invoke PEFT generation."""
    generator, tokenizer, model = build_generator(
        monkeypatch,
        tmp_path,
    )

    generation = generator.generate(
        build_request()
    )

    assert generation.answer == "Hold the invoice."
    assert generation.cited_document_ids == (
        "duplicate-policy",
    )
    assert generation.abstained is False

    assert model.generate_kwargs is not None
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 64
    assert model.generate_kwargs["pad_token_id"] == 99

    assert tokenizer.messages is not None
    assert (
        "duplicate-policy"
        in tokenizer.messages[1]["content"]
    )


def test_peft_generator_abstains_for_weak_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Weak retrieval evidence should bypass generation."""
    generator, _, model = build_generator(
        monkeypatch,
        tmp_path,
    )

    generation = generator.generate(
        build_request(
            score=0.3,
        )
    )

    assert generation.abstained is True
    assert generation.cited_document_ids == ()
    assert model.generate_kwargs is None


def test_peft_generator_supports_model_abstention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit insufficient-evidence output should abstain."""
    generator, _, _ = build_generator(
        monkeypatch,
        tmp_path,
        decoded_answer="INSUFFICIENT_EVIDENCE",
    )

    generation = generator.generate(
        build_request()
    )

    assert generation.answer == "Insufficient evidence."
    assert generation.cited_document_ids == ()
    assert generation.abstained is True


def test_peft_generator_abstains_for_empty_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Empty decoded output should become an abstention."""
    generator, _, _ = build_generator(
        monkeypatch,
        tmp_path,
        decoded_answer="   ",
    )

    generation = generator.generate(
        build_request()
    )

    assert generation.abstained is True
    assert generation.cited_document_ids == ()


def test_peft_generator_rejects_missing_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A nonexistent adapter directory must fail before loading."""
    tokenizer = FakeTokenizer()
    model = FakeModel()

    patch_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    with pytest.raises(
        ValueError,
        match="adapter_path",
    ):
        HuggingFacePeftCausalGenerator(
            model_id="example/model",
            revision="abc123",
            adapter_path=tmp_path / "missing",
        )


def test_peft_generator_rejects_invalid_generation_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid generation configuration should fail early."""
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()

    tokenizer = FakeTokenizer()
    model = FakeModel()

    patch_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    with pytest.raises(
        ValueError,
        match="max_new_tokens",
    ):
        HuggingFacePeftCausalGenerator(
            model_id="example/model",
            revision="abc123",
            adapter_path=adapter_path,
            max_new_tokens=0,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("model_id", "", "model_id"),
        ("revision", "", "revision"),
        ("device", "", "device"),
        ("minimum_score", 2.0, "minimum_score"),
    ],
)
def test_peft_generator_rejects_invalid_core_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    """Invalid core configuration should fail before model use."""
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()

    tokenizer = FakeTokenizer()
    model = FakeModel()

    patch_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    kwargs: dict[str, object] = {
        "model_id": "example/model",
        "revision": "abc123",
        "adapter_path": adapter_path,
        "device": "cpu",
        "minimum_score": 0.4,
    }

    kwargs[field] = value

    with pytest.raises(
        ValueError,
        match=match,
    ):
        HuggingFacePeftCausalGenerator(
            **kwargs,  # type: ignore[arg-type]
        )


def test_peft_generator_abstains_without_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generation without retrieved evidence should abstain."""
    generator, _, model = build_generator(
        monkeypatch,
        tmp_path,
    )

    generation = generator.generate(
        RAGGenerationRequest(
            query="What should happen?",
            context_documents=(),
        )
    )

    assert generation.abstained is True
    assert generation.cited_document_ids == ()
    assert model.generate_kwargs is None
