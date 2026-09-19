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
    pad_token_id = 99
    padding_side = "right"

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
        prompt: str | list[str],
        *,
        return_tensors: str,
        padding: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return deterministic single or batch prompt tokens."""
        assert return_tensors == "pt"

        if isinstance(
            prompt,
            str,
        ):
            assert prompt == "portable prompt"
            assert padding is False

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

        assert prompt
        assert all(
            item == "portable prompt"
            for item in prompt
        )
        assert padding is True

        batch_size = len(
            prompt
        )

        return {
            "input_ids": torch.tensor(
                [
                    [1, 2, 3]
                    for _ in range(
                        batch_size
                    )
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1]
                    for _ in range(
                        batch_size
                    )
                ],
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
        self.generate_call_count = 0

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
        self.generate_call_count += 1

        if self.error is not None:
            raise self.error

        input_ids = kwargs["input_ids"]

        batch_size = int(
            input_ids.shape[0]
        )

        generated = torch.tensor(
            [
                [7, 8]
                for _ in range(
                    batch_size
                )
            ],
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


def test_local_provider_executes_one_genuine_native_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native batching must use exactly one model.generate invocation."""
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

    requests = (
        build_request(),
        build_request().model_copy(
            update={
                "request_id": "request-2",
            }
        ),
    )

    responses = provider.generate_batch(
        requests
    )

    assert len(
        responses
    ) == 2

    assert (
        model.generate_call_count
        == 1
    )

    assert (
        tokenizer.padding_side
        == "left"
    )

    assert [
        response.request_id
        for response
        in responses
    ] == [
        "request-1",
        "request-2",
    ]

    assert all(
        response.succeeded
        for response
        in responses
    )

    assert all(
        response.output_text
        == "verified response"
        for response
        in responses
    )

    assert all(
        response.usage
        is not None
        for response
        in responses
    )

    assert [
        response.usage.input_tokens
        if response.usage
        is not None
        else None
        for response
        in responses
    ] == [
        3,
        3,
    ]

    assert [
        response.usage.output_tokens
        if response.usage
        is not None
        else None
        for response
        in responses
    ] == [
        2,
        2,
    ]

    assert all(
        response.metadata[
            "execution_mode"
        ]
        == "native-batch"
        for response
        in responses
    )

    assert all(
        response.metadata[
            "native_batch_size"
        ]
        == 2
        for response
        in responses
    )

    assert model.generate_kwargs is not None

    assert (
        model.generate_kwargs[
            "input_ids"
        ].shape[0]
        == 2
    )


def test_local_provider_rejects_empty_native_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty batch is an invalid provider invocation."""
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

    with pytest.raises(
        ValueError,
        match="At least one provider request",
    ):
        provider.generate_batch(
            ()
        )

    assert (
        model.generate_call_count
        == 0
    )


def test_local_provider_rejects_incompatible_native_batch_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One native generation call requires homogeneous generation settings."""
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

    requests = (
        build_request(),
        build_request().model_copy(
            update={
                "request_id": "request-2",
                "max_output_tokens": 32,
            }
        ),
    )

    responses = provider.generate_batch(
        requests
    )

    assert (
        model.generate_call_count
        == 0
    )

    assert len(
        responses
    ) == 2

    assert all(
        not response.succeeded
        for response
        in responses
    )

    assert all(
        response.error
        is not None
        for response
        in responses
    )

    assert all(
        response.error.error_type
        == "batch_configuration_mismatch"
        for response
        in responses
        if response.error
        is not None
    )


def test_local_provider_normalises_native_batch_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One backend batch failure must become per-request error evidence."""
    tokenizer = FakeTokenizer()

    model = FakeModel(
        error=RuntimeError(
            "native batch generation failed"
        ),
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

    responses = provider.generate_batch(
        (
            build_request(),
            build_request().model_copy(
                update={
                    "request_id": "request-2",
                }
            ),
        )
    )

    assert (
        model.generate_call_count
        == 1
    )

    assert len(
        responses
    ) == 2

    assert all(
        not response.succeeded
        for response
        in responses
    )

    assert all(
        response.error
        is not None
        and response.error.error_type
        == "RuntimeError"
        and response.error.message
        == "native batch generation failed"
        for response
        in responses
    )


def test_local_provider_native_batch_rejects_model_mismatch_without_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed-model batch must never enter the pinned local model."""
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

    responses = provider.generate_batch(
        (
            build_request(),
            build_request(
                model_id="different/model"
            ).model_copy(
                update={
                    "request_id": "request-2",
                }
            ),
        )
    )

    assert (
        model.generate_call_count
        == 0
    )

    assert all(
        not response.succeeded
        for response
        in responses
    )

    assert all(
        response.error
        is not None
        and response.error.error_type
        == "model_mismatch"
        for response
        in responses
    )
