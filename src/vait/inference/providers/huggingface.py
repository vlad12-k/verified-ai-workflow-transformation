"""Local Hugging Face implementation of the provider-neutral contract."""

from time import perf_counter_ns
from typing import Any, Protocol, Self, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vait.inference.providers.models import (
    ProviderError,
    ProviderInferenceRequest,
    ProviderInferenceResponse,
    ProviderTiming,
    ProviderTokenUsage,
)


class _LocalTokenizer(Protocol):
    """Typed boundary for tokenizer operations used by the adapter."""

    eos_token_id: int | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Render provider-neutral messages into one model prompt."""
        ...

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        """Tokenize one rendered prompt."""
        ...

    def decode(
        self,
        tokens: torch.Tensor,
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Decode generated token IDs."""
        ...


class _LocalCausalModel(Protocol):
    """Typed boundary for local causal-model operations."""

    def to(
        self,
        device: str,
    ) -> Self:
        """Move the model to the requested device."""
        ...

    def eval(self) -> Self:
        """Switch the model to evaluation mode."""
        ...

    def generate(
        self,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate token IDs."""
        ...


class LocalHuggingFaceProvider:
    """Serve one pinned local Hugging Face causal language model."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> None:
        """Load one pinned model behind the portable provider boundary."""
        if not model_id.strip():
            raise ValueError("model_id must not be empty.")

        if not revision.strip():
            raise ValueError("revision must not be empty.")

        if not device.strip():
            raise ValueError("device must not be empty.")

        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._local_files_only = local_files_only

        loaded_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        loaded_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )

        self._tokenizer = cast(
            _LocalTokenizer,
            loaded_tokenizer,
        )
        self._model = cast(
            _LocalCausalModel,
            loaded_model,
        )

        self._model.to(device)
        self._model.eval()

    @property
    def provider_id(self) -> str:
        """Return stable provider identity."""
        return "local-huggingface"

    @property
    def runtime_id(self) -> str:
        """Return stable runtime identity."""
        return "transformers-causal-lm"

    @property
    def model_id(self) -> str:
        """Return the pinned local model identifier."""
        return self._model_id

    @property
    def revision(self) -> str:
        """Return the pinned model revision."""
        return self._revision

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Execute one local generative request with normalised evidence."""
        started = perf_counter_ns()

        if request.model_id != self._model_id:
            return self._error_response(
                request=request,
                started=started,
                error_type="model_mismatch",
                message=(
                    f"Provider serves '{self._model_id}', "
                    f"not '{request.model_id}'."
                ),
            )

        try:
            messages = [
                {
                    "role": message.role.value,
                    "content": message.content,
                }
                for message in request.messages
            ]

            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            encoded = self._tokenizer(
                prompt,
                return_tensors="pt",
            )

            model_inputs = {
                name: tensor.to(self._device)
                for name, tensor in encoded.items()
            }

            input_tokens = int(
                model_inputs["input_ids"].shape[1]
            )

            generation_kwargs: dict[str, Any] = {
                **model_inputs,
                "max_new_tokens": request.max_output_tokens,
            }

            if request.temperature > 0.0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = (
                    request.temperature
                )
            else:
                generation_kwargs["do_sample"] = False

            if self._tokenizer.eos_token_id is not None:
                generation_kwargs["pad_token_id"] = (
                    self._tokenizer.eos_token_id
                )

            generation_started = perf_counter_ns()

            with torch.inference_mode():
                output = self._model.generate(
                    **generation_kwargs
                )

            generation_latency_ms = (
                perf_counter_ns() - generation_started
            ) / 1_000_000

            generated_tokens = output[
                0,
                input_tokens:,
            ]

            output_tokens = int(
                generated_tokens.numel()
            )

            output_text = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            if not isinstance(output_text, str):
                raise TypeError(
                    "Tokenizer decode must return a string."
                )

            total_latency_ms = (
                perf_counter_ns() - started
            ) / 1_000_000

            return ProviderInferenceResponse(
                request_id=request.request_id,
                provider_id=self.provider_id,
                runtime_id=self.runtime_id,
                model_id=self._model_id,
                model_revision=self._revision,
                output_text=output_text.strip(),
                usage=ProviderTokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                timing=ProviderTiming(
                    total_latency_ms=total_latency_ms,
                    time_to_first_token_ms=None,
                    generation_latency_ms=(
                        generation_latency_ms
                    ),
                ),
                cost_usd=None,
                metadata={
                    "device": self._device,
                    "local_files_only": (
                        self._local_files_only
                    ),
                    "do_sample": (
                        request.temperature > 0.0
                    ),
                    "ttft_available": False,
                },
            )

        except Exception as exc:
            return self._error_response(
                request=request,
                started=started,
                error_type=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
            )

    def _error_response(
        self,
        *,
        request: ProviderInferenceRequest,
        started: int,
        error_type: str,
        message: str,
    ) -> ProviderInferenceResponse:
        """Return one normalised local-provider failure."""
        total_latency_ms = (
            perf_counter_ns() - started
        ) / 1_000_000

        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=self._model_id,
            model_revision=self._revision,
            timing=ProviderTiming(
                total_latency_ms=total_latency_ms,
            ),
            error=ProviderError(
                error_type=error_type,
                message=message,
                retryable=False,
            ),
            metadata={
                "device": self._device,
                "local_files_only": (
                    self._local_files_only
                ),
            },
        )
