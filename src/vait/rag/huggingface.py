"""Local Hugging Face causal-language-model RAG generator."""

from typing import Any, Protocol, Self, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vait.rag.models import (
    RAGGeneration,
    RAGGenerationRequest,
)
from vait.rag.prompt_contracts import (
    INSUFFICIENT_EVIDENCE,
    RAGPromptContract,
    build_rag_messages,
)


class _LocalCausalModel(Protocol):
    """Typed boundary for the causal-model operations VAIT uses."""

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
        """Generate token IDs from encoded model inputs."""
        ...


class HuggingFaceCausalGenerator:
    """Generate grounded answers with a pinned local causal language model."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        device: str = "cpu",
        minimum_score: float = 0.4,
        max_new_tokens: int = 80,
        local_files_only: bool = True,
        prompt_contract: RAGPromptContract = (
            RAGPromptContract.BASELINE_V1
        ),
    ) -> None:
        """Load a reproducible causal model and deterministic tokenizer."""
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

        if not -1.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between -1.0 and 1.0."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._minimum_score = minimum_score
        self._max_new_tokens = max_new_tokens
        self._local_files_only = local_files_only
        self._prompt_contract = prompt_contract

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )

        loaded_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )

        self._model = cast(
            _LocalCausalModel,
            loaded_model,
        )
        self._model.to(device)
        self._model.eval()

    @property
    def implementation_id(self) -> str:
        """Return the pinned model and inference configuration."""
        return (
            f"huggingface-causal:{self._model_id}"
            f"@{self._revision}:"
            f"device={self._device}:"
            f"minimum-score={self._minimum_score:.6f}:"
            f"max-new-tokens={self._max_new_tokens}:"
            f"prompt-contract={self._prompt_contract.value}"
        )

    @property
    def model_id(self) -> str:
        """Return the Hugging Face model identifier."""
        return self._model_id

    @property
    def revision(self) -> str:
        """Return the pinned model revision."""
        return self._revision

    @property
    def device(self) -> str:
        """Return the inference device."""
        return self._device

    @property
    def minimum_score(self) -> float:
        """Return the retrieval abstention threshold."""
        return self._minimum_score

    @property
    def prompt_contract(self) -> RAGPromptContract:
        """Return the versioned generation contract."""
        return self._prompt_contract

    def generate(
        self,
        request: RAGGenerationRequest,
    ) -> RAGGeneration:
        """Generate from supplied evidence or explicitly abstain."""
        if not request.context_documents:
            return self._abstain()

        ordered_context = tuple(
            sorted(
                request.context_documents,
                key=lambda document: document.rank,
            )
        )

        strongest_document = ordered_context[0]

        if strongest_document.score < self._minimum_score:
            return self._abstain()

        messages = build_rag_messages(
            request,
            context_documents=ordered_context,
            contract=self._prompt_contract,
        )

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

        prompt_length = int(
            model_inputs["input_ids"].shape[1]
        )

        with torch.inference_mode():
            output = self._model.generate(
                **model_inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated_tokens = output[
            0,
            prompt_length:,
        ]

        decoded_answer = self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        if not isinstance(decoded_answer, str):
            raise TypeError(
                "Tokenizer decode must return a string "
                "for one generated sequence."
            )

        answer = decoded_answer.strip()

        if not answer:
            return self._abstain()

        if answer == INSUFFICIENT_EVIDENCE:
            return self._abstain()

        return RAGGeneration(
            answer=answer,
            cited_document_ids=tuple(
                document.document_id
                for document in ordered_context
            ),
            abstained=False,
        )

    @staticmethod
    def _abstain() -> RAGGeneration:
        """Return a deterministic no-evidence result."""
        return RAGGeneration(
            answer="Insufficient evidence.",
            cited_document_ids=(),
            abstained=True,
        )
