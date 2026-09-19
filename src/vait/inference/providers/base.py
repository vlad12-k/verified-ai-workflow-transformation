"""Provider-neutral inference protocol."""

from collections.abc import Sequence
from typing import Protocol

from vait.inference.providers.models import (
    ProviderInferenceRequest,
    ProviderInferenceResponse,
)


class InferenceProvider(Protocol):
    """Portable boundary implemented by concrete inference providers."""

    @property
    def provider_id(self) -> str:
        """Return stable provider identity."""
        ...

    @property
    def runtime_id(self) -> str:
        """Return stable runtime identity."""
        ...

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Execute one provider-neutral generative inference request."""
        ...


class BatchInferenceProvider(
    InferenceProvider,
    Protocol,
):
    """Optional provider capability for genuine native batch inference."""

    def generate_batch(
        self,
        requests: Sequence[
            ProviderInferenceRequest
        ],
    ) -> tuple[
        ProviderInferenceResponse,
        ...,
    ]:
        """Execute one native batch of independent generative requests."""
        ...
