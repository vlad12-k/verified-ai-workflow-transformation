"""Tests for the local Hugging Face inference provider."""

from typing import Any

import pytest
import torch

from vait.inference.providers.huggingface import (
    LocalHuggingFaceProvider,
)
from vait.inference.providers.models import (
    InferenceMessage,
    InferenceRole,
    ProviderInferenceRequest,
)


class FakeTokenizer:
    """Minimal deterministic tokenizer double."""

    eos_token_id = 99

    def __init__(self) -> None:
        """Track rendered messages."""
        self.messages: list[dict[str, str]] | None = None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Return one deterministic prompt."""
        assert tokenize is False
        assert add_generation_prompt is True

        self.messages = conversation

        return "portable prompt"

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        """Return three deterministic prompt tokens."""
        assert prompt == "portable prompt"
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
        """Decode two deterministic generated tokens."""
        assert tokens.numel() == 2
        assert skip_special_tokens is True

        return "verified response"


class FakeModel:
    """Minimal deterministic causal-model double."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
    ) -> None:
        """Track model state and optional generation failure."""
        self.error = error
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
        """Append two generated token IDs or raise the configured failure."""
        self.generate_kwargs = kwargs

        if self.error is not None:
            raise self.error

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


def patch_loading(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer: FakeTokenizer,
    model: FakeModel,
) -> None:
    """Replace Hugging Face loading with deterministic doubles."""
    monkeypatch.setattr(
        "vait.inference.providers.huggingface."
        "AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "vait.inference.providers.huggingface."
        "AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )


def build_request(
    *,
    model_id: str = "example/model",
    temperature: float = 0.0,
) -> ProviderInferenceRequest:
    """Build one portable provider request."""
    return ProviderInferenceRequest(
        request_id="request-1",
        model_id=model_id,
        messages=(
            InferenceMessage(
                role=InferenceRole.SYSTEM,
                content="Use supplied evidence only.",
            ),
            InferenceMessage(
                role=InferenceRole.USER,
                content="What is the verified result?",
            ),
        ),
        max_output_tokens=16,
        temperature=temperature,
    )


def test_local_provider_normalises_successful_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local generation should produce portable usage and timing evidence."""
    tokenizer = FakeTokenizer()
    model = FakeModel()

    patch_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    provider = LocalHuggingFaceProvider(
        model_id="example/model",
        revision="abc123",
        device="cpu",
        local_files_only=True,
    )

    response = provider.generate(
        build_request()
    )

    assert response.succeeded is True

    assert response.provider_id == "local-huggingface"
    assert response.runtime_id == "transformers-causal-lm"

    assert response.model_id == "example/model"
    assert response.model_revision == "abc123"
    assert response.output_text == "verified response"

    assert response.usage is not None
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 2

    assert response.timing.total_latency_ms >= 0.0
    assert response.timing.time_to_first_token_ms is None

    assert response.timing.generation_latency_ms is not None
    assert response.timing.generation_latency_ms >= 0.0

    assert response.cost_usd is None

    assert response.metadata["device"] == "cpu"
    assert response.metadata["ttft_available"] is False

    assert model.device == "cpu"
    assert model.eval_called is True

    assert model.generate_kwargs is not None
    assert model.generate_kwargs["max_new_tokens"] == 16
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["pad_token_id"] == 99

    assert tokenizer.messages is not None
    assert tokenizer.messages[0]["role"] == "system"
    assert tokenizer.messages[1]["role"] == "user"


def test_local_provider_maps_sampling_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive temperature should map to explicit local sampling."""
    tokenizer = FakeTokenizer()
    model = FakeModel()

    patch_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    provider = LocalHuggingFaceProvider(
        model_id="example/model",
        revision="abc123",
    )

    response = provider.generate(
        build_request(
            temperature=0.7,
        )
    )

    assert response.succeeded is True
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["do_sample"] is True
    assert model.generate_kwargs["temperature"] == 0.7


def test_local_provider_normalises_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend exceptions should become provider-neutral error evidence."""
    tokenizer = FakeTokenizer()
    model = FakeModel(
        error=RuntimeError("local generation failed"),
    )

    patch_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    provider = LocalHuggingFaceProvider(
        model_id="example/model",
        revision="abc123",
    )

    response = provider.generate(
        build_request()
    )

    assert response.succeeded is False
    assert response.output_text is None
    assert response.usage is None

    assert response.error is not None
    assert response.error.error_type == "RuntimeError"
    assert response.error.message == "local generation failed"
    assert response.error.retryable is False

    assert response.timing.total_latency_ms >= 0.0


def test_local_provider_rejects_model_mismatch_without_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pinned provider must not silently execute a different model."""
    tokenizer = FakeTokenizer()
    model = FakeModel()

    patch_loading(
        monkeypatch,
        tokenizer=tokenizer,
        model=model,
    )

    provider = LocalHuggingFaceProvider(
        model_id="example/model",
        revision="abc123",
    )

    response = provider.generate(
        build_request(
            model_id="different/model",
        )
    )

    assert response.succeeded is False

    assert response.error is not None
    assert response.error.error_type == "model_mismatch"

    assert model.generate_kwargs is None
