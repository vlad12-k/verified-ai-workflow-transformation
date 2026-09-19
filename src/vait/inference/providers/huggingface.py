"""Local Hugging Face implementation of the provider-neutral contract."""

from collections.abc import Sequence
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
    pad_token_id: int | None
    padding_side: str

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
        prompt: str | Sequence[str],
        *,
        return_tensors: str,
        padding: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Tokenize one prompt or one padded prompt batch."""
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
            raise ValueError(
                "model_id must not be empty."
            )

        if not revision.strip():
            raise ValueError(
                "revision must not be empty."
            )

        if not device.strip():
            raise ValueError(
                "device must not be empty."
            )

        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._local_files_only = local_files_only

        loaded_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )

        if (
            getattr(
                loaded_tokenizer,
                "pad_token_id",
                None,
            )
            is None
        ):
            eos_token = getattr(
                loaded_tokenizer,
                "eos_token",
                None,
            )

            if eos_token is None:
                raise ValueError(
                    "Local causal tokenizer requires "
                    "a pad token or EOS token for "
                    "native batching."
                )

            loaded_tokenizer.pad_token = eos_token

        loaded_tokenizer.padding_side = "left"

        loaded_model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_id,
                revision=revision,
                local_files_only=local_files_only,
            )
        )

        self._tokenizer = cast(
            _LocalTokenizer,
            loaded_tokenizer,
        )

        self._model = cast(
            _LocalCausalModel,
            loaded_model,
        )

        self._model.to(
            device
        )

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
            messages = self._request_messages(
                request
            )

            prompt = (
                self._tokenizer
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            encoded = self._tokenizer(
                prompt,
                return_tensors="pt",
            )

            model_inputs = {
                name: tensor.to(
                    self._device
                )
                for name, tensor
                in encoded.items()
            }

            input_tokens = int(
                model_inputs[
                    "input_ids"
                ].shape[1]
            )

            generation_kwargs = (
                self._generation_kwargs(
                    model_inputs=(
                        model_inputs
                    ),
                    max_output_tokens=(
                        request
                        .max_output_tokens
                    ),
                    temperature=(
                        request.temperature
                    ),
                )
            )

            generation_started = (
                perf_counter_ns()
            )

            with torch.inference_mode():
                output = (
                    self._model.generate(
                        **generation_kwargs
                    )
                )

            generation_latency_ms = (
                perf_counter_ns()
                - generation_started
            ) / 1_000_000

            generated_tokens = output[
                0,
                input_tokens:,
            ]

            output_tokens = int(
                generated_tokens.numel()
            )

            output_text = (
                self._tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )
            )

            if not isinstance(
                output_text,
                str,
            ):
                raise TypeError(
                    "Tokenizer decode must "
                    "return a string."
                )

            total_latency_ms = (
                perf_counter_ns()
                - started
            ) / 1_000_000

            return (
                ProviderInferenceResponse(
                    request_id=(
                        request.request_id
                    ),
                    provider_id=(
                        self.provider_id
                    ),
                    runtime_id=(
                        self.runtime_id
                    ),
                    model_id=(
                        self._model_id
                    ),
                    model_revision=(
                        self._revision
                    ),
                    output_text=(
                        output_text.strip()
                    ),
                    usage=(
                        ProviderTokenUsage(
                            input_tokens=(
                                input_tokens
                            ),
                            output_tokens=(
                                output_tokens
                            ),
                        )
                    ),
                    timing=ProviderTiming(
                        total_latency_ms=(
                            total_latency_ms
                        ),
                        time_to_first_token_ms=None,
                        generation_latency_ms=(
                            generation_latency_ms
                        ),
                    ),
                    cost_usd=None,
                    metadata={
                        "device": (
                            self._device
                        ),
                        "local_files_only": (
                            self
                            ._local_files_only
                        ),
                        "do_sample": (
                            request.temperature
                            > 0.0
                        ),
                        "ttft_available": (
                            False
                        ),
                        "execution_mode": (
                            "single-request"
                        ),
                    },
                )
            )

        except Exception as exc:
            return self._error_response(
                request=request,
                started=started,
                error_type=(
                    type(exc).__name__
                ),
                message=(
                    str(exc)
                    or type(exc).__name__
                ),
            )

    def generate_batch(
        self,
        requests: Sequence[
            ProviderInferenceRequest
        ],
    ) -> tuple[
        ProviderInferenceResponse,
        ...,
    ]:
        """Execute one genuine native batch through one model.generate call."""
        if not requests:
            raise ValueError(
                "At least one provider request "
                "is required for native batching."
            )

        batch_started = (
            perf_counter_ns()
        )

        request_tuple = tuple(
            requests
        )

        if any(
            request.model_id
            != self._model_id
            for request
            in request_tuple
        ):
            return tuple(
                self._error_response(
                    request=request,
                    started=(
                        batch_started
                    ),
                    error_type=(
                        "model_mismatch"
                    ),
                    message=(
                        f"Provider serves "
                        f"'{self._model_id}', "
                        f"not "
                        f"'{request.model_id}'."
                    ),
                )
                for request
                in request_tuple
            )

        max_output_tokens = (
            request_tuple[
                0
            ].max_output_tokens
        )

        temperature = (
            request_tuple[
                0
            ].temperature
        )

        if any(
            request.max_output_tokens
            != max_output_tokens
            or request.temperature
            != temperature
            for request
            in request_tuple
        ):
            return tuple(
                self._error_response(
                    request=request,
                    started=(
                        batch_started
                    ),
                    error_type=(
                        "batch_configuration_mismatch"
                    ),
                    message=(
                        "Native batch requests "
                        "must use identical "
                        "max_output_tokens and "
                        "temperature."
                    ),
                )
                for request
                in request_tuple
            )

        try:
            prompts = [
                self._tokenizer
                .apply_chat_template(
                    self._request_messages(
                        request
                    ),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for request
                in request_tuple
            ]

            encoded = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            )

            model_inputs = {
                name: tensor.to(
                    self._device
                )
                for name, tensor
                in encoded.items()
            }

            input_ids = model_inputs[
                "input_ids"
            ]

            prompt_width = int(
                input_ids.shape[1]
            )

            attention_mask = (
                model_inputs.get(
                    "attention_mask"
                )
            )

            if attention_mask is None:
                input_token_counts = [
                    prompt_width
                    for _ in request_tuple
                ]
            else:
                input_token_counts = [
                    int(
                        value.item()
                    )
                    for value
                    in attention_mask.sum(
                        dim=1
                    )
                ]

            generation_kwargs = (
                self._generation_kwargs(
                    model_inputs=(
                        model_inputs
                    ),
                    max_output_tokens=(
                        max_output_tokens
                    ),
                    temperature=(
                        temperature
                    ),
                )
            )

            generation_started = (
                perf_counter_ns()
            )

            with torch.inference_mode():
                output = (
                    self._model.generate(
                        **generation_kwargs
                    )
                )

            generation_latency_ms = (
                perf_counter_ns()
                - generation_started
            ) / 1_000_000

            if int(
                output.shape[0]
            ) != len(
                request_tuple
            ):
                raise RuntimeError(
                    "Native batch generation "
                    "returned an unexpected "
                    "batch dimension."
                )

            generated_matrix = output[
                :,
                prompt_width:,
            ]

            total_latency_ms = (
                perf_counter_ns()
                - batch_started
            ) / 1_000_000

            responses: list[
                ProviderInferenceResponse
            ] = []

            for (
                index,
                request,
            ) in enumerate(
                request_tuple
            ):
                generated_tokens = (
                    self
                    ._trim_generated_tokens(
                        generated_matrix[
                            index
                        ]
                    )
                )

                output_tokens = int(
                    generated_tokens.numel()
                )

                output_text = (
                    self._tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True,
                    )
                )

                if not isinstance(
                    output_text,
                    str,
                ):
                    raise TypeError(
                        "Tokenizer decode "
                        "must return a string."
                    )

                responses.append(
                    ProviderInferenceResponse(
                        request_id=(
                            request
                            .request_id
                        ),
                        provider_id=(
                            self.provider_id
                        ),
                        runtime_id=(
                            self.runtime_id
                        ),
                        model_id=(
                            self._model_id
                        ),
                        model_revision=(
                            self._revision
                        ),
                        output_text=(
                            output_text
                            .strip()
                        ),
                        usage=(
                            ProviderTokenUsage(
                                input_tokens=(
                                    input_token_counts[
                                        index
                                    ]
                                ),
                                output_tokens=(
                                    output_tokens
                                ),
                            )
                        ),
                        timing=(
                            ProviderTiming(
                                total_latency_ms=(
                                    total_latency_ms
                                ),
                                time_to_first_token_ms=None,
                                generation_latency_ms=(
                                    generation_latency_ms
                                ),
                            )
                        ),
                        cost_usd=None,
                        metadata={
                            "device": (
                                self._device
                            ),
                            "local_files_only": (
                                self
                                ._local_files_only
                            ),
                            "do_sample": (
                                temperature
                                > 0.0
                            ),
                            "ttft_available": (
                                False
                            ),
                            "execution_mode": (
                                "native-batch"
                            ),
                            "native_batch_size": (
                                len(
                                    request_tuple
                                )
                            ),
                            "batch_latency_shared": (
                                True
                            ),
                            "padding_side": (
                                self._tokenizer
                                .padding_side
                            ),
                        },
                    )
                )

            return tuple(
                responses
            )

        except Exception as exc:
            return tuple(
                self._error_response(
                    request=request,
                    started=(
                        batch_started
                    ),
                    error_type=(
                        type(exc).__name__
                    ),
                    message=(
                        str(exc)
                        or type(exc).__name__
                    ),
                )
                for request
                in request_tuple
            )

    @staticmethod
    def _request_messages(
        request: ProviderInferenceRequest,
    ) -> list[
        dict[str, str]
    ]:
        """Convert provider-neutral messages for the local tokenizer."""
        return [
            {
                "role": (
                    message.role.value
                ),
                "content": (
                    message.content
                ),
            }
            for message
            in request.messages
        ]

    def _generation_kwargs(
        self,
        *,
        model_inputs: dict[
            str,
            torch.Tensor,
        ],
        max_output_tokens: int,
        temperature: float,
    ) -> dict[
        str,
        Any,
    ]:
        """Build deterministic or sampled local generation arguments."""
        generation_kwargs: dict[
            str,
            Any,
        ] = {
            **model_inputs,
            "max_new_tokens": (
                max_output_tokens
            ),
        }

        if temperature > 0.0:
            generation_kwargs[
                "do_sample"
            ] = True

            generation_kwargs[
                "temperature"
            ] = temperature
        else:
            generation_kwargs[
                "do_sample"
            ] = False

        if (
            self._tokenizer
            .eos_token_id
            is not None
        ):
            generation_kwargs[
                "pad_token_id"
            ] = (
                self._tokenizer
                .eos_token_id
            )

        return generation_kwargs

    def _trim_generated_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Trim batch padding while preserving the first EOS token."""
        eos_token_id = (
            self._tokenizer
            .eos_token_id
        )

        if eos_token_id is None:
            return tokens

        eos_positions = (
            tokens
            .eq(
                eos_token_id
            )
            .nonzero(
                as_tuple=False
            )
        )

        if (
            eos_positions.numel()
            == 0
        ):
            return tokens

        first_eos_index = int(
            eos_positions[
                0,
                0,
            ].item()
        )

        return tokens[
            : first_eos_index + 1
        ]

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
            perf_counter_ns()
            - started
        ) / 1_000_000

        return (
            ProviderInferenceResponse(
                request_id=(
                    request.request_id
                ),
                provider_id=(
                    self.provider_id
                ),
                runtime_id=(
                    self.runtime_id
                ),
                model_id=(
                    self._model_id
                ),
                model_revision=(
                    self._revision
                ),
                timing=ProviderTiming(
                    total_latency_ms=(
                        total_latency_ms
                    ),
                ),
                error=ProviderError(
                    error_type=(
                        error_type
                    ),
                    message=message,
                    retryable=False,
                ),
                metadata={
                    "device": (
                        self._device
                    ),
                    "local_files_only": (
                        self
                        ._local_files_only
                    ),
                },
            )
        )
