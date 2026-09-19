"""Provider-neutral deterministic response caching."""

import json
from hashlib import sha256
from threading import Lock
from time import perf_counter_ns

from vait.inference.providers.base import InferenceProvider
from vait.inference.providers.models import (
    ProviderInferenceRequest,
    ProviderInferenceResponse,
    ProviderTiming,
    ProviderTokenUsage,
)


class DeterministicCachingProvider:
    """Cache successful deterministic provider responses by semantic request."""

    def __init__(
        self,
        *,
        provider: InferenceProvider,
    ) -> None:
        """Wrap one provider with an in-memory deterministic response cache."""
        self._provider = provider
        self._cache: dict[
            str,
            ProviderInferenceResponse,
        ] = {}

        self._lock = Lock()

        self._hits = 0
        self._misses = 0
        self._bypasses = 0

    @property
    def provider_id(self) -> str:
        """Return the wrapped provider identity."""
        return self._provider.provider_id

    @property
    def runtime_id(self) -> str:
        """Return a distinct cached runtime identity."""
        return (
            f"{self._provider.runtime_id}"
            "+deterministic-response-cache-v1"
        )

    @property
    def cache_size(self) -> int:
        """Return the number of cached semantic requests."""
        with self._lock:
            return len(
                self._cache
            )

    @property
    def cache_hits(self) -> int:
        """Return successful cache-hit count."""
        with self._lock:
            return self._hits

    @property
    def cache_misses(self) -> int:
        """Return cache-miss count."""
        with self._lock:
            return self._misses

    @property
    def cache_bypasses(self) -> int:
        """Return requests excluded from deterministic caching."""
        with self._lock:
            return self._bypasses

    def clear(self) -> None:
        """Clear cached responses without rewriting historical statistics."""
        with self._lock:
            self._cache.clear()

    def generate(
        self,
        request: ProviderInferenceRequest,
    ) -> ProviderInferenceResponse:
        """Serve one request from cache or the wrapped provider."""
        started = perf_counter_ns()

        if request.temperature != 0.0:
            with self._lock:
                self._bypasses += 1

            response = self._provider.generate(
                request
            )

            self._validate_request_identity(
                request=request,
                response=response,
            )

            return self._decorate_passthrough(
                response=response,
                status="bypass",
                cache_key=None,
                cache_eligible=False,
                cache_stored=False,
            )

        cache_key = self._cache_key(
            request
        )

        with self._lock:
            cached = self._cache.get(
                cache_key
            )

            if cached is not None:
                self._hits += 1

        if cached is not None:
            return self._cache_hit_response(
                request=request,
                cached=cached,
                cache_key=cache_key,
                started=started,
            )

        with self._lock:
            self._misses += 1

        response = self._provider.generate(
            request
        )

        self._validate_request_identity(
            request=request,
            response=response,
        )

        stored = False

        if response.succeeded:
            with self._lock:
                self._cache.setdefault(
                    cache_key,
                    response,
                )

            stored = True

        return self._decorate_passthrough(
            response=response,
            status="miss",
            cache_key=cache_key,
            cache_eligible=True,
            cache_stored=stored,
        )

    def _cache_hit_response(
        self,
        *,
        request: ProviderInferenceRequest,
        cached: ProviderInferenceResponse,
        cache_key: str,
        started: int,
    ) -> ProviderInferenceResponse:
        """Build one zero-provider-work cache-hit response."""
        lookup_latency_ms = (
            perf_counter_ns()
            - started
        ) / 1_000_000

        origin_input_tokens = (
            cached.usage.input_tokens
            if cached.usage
            is not None
            else None
        )

        origin_output_tokens = (
            cached.usage.output_tokens
            if cached.usage
            is not None
            else None
        )

        return ProviderInferenceResponse(
            request_id=request.request_id,
            provider_id=self.provider_id,
            runtime_id=self.runtime_id,
            model_id=cached.model_id,
            model_revision=(
                cached.model_revision
            ),
            output_text=(
                cached.output_text
            ),
            usage=ProviderTokenUsage(
                input_tokens=0,
                output_tokens=0,
            ),
            timing=ProviderTiming(
                total_latency_ms=(
                    lookup_latency_ms
                ),
                time_to_first_token_ms=None,
                generation_latency_ms=None,
            ),
            cost_usd=0.0,
            error=None,
            metadata={
                **cached.metadata,
                "cache_status": "hit",
                "cache_eligible": True,
                "cache_stored": True,
                "cache_key_sha256": (
                    cache_key
                ),
                "cache_origin_request_id": (
                    cached.request_id
                ),
                "cache_origin_input_tokens": (
                    origin_input_tokens
                ),
                "cache_origin_output_tokens": (
                    origin_output_tokens
                ),
                "cache_lookup_latency_ms": (
                    lookup_latency_ms
                ),
            },
        )

    def _decorate_passthrough(
        self,
        *,
        response: ProviderInferenceResponse,
        status: str,
        cache_key: str | None,
        cache_eligible: bool,
        cache_stored: bool,
    ) -> ProviderInferenceResponse:
        """Attach cache provenance without changing provider evidence."""
        metadata = {
            **response.metadata,
            "cache_status": status,
            "cache_eligible": (
                cache_eligible
            ),
            "cache_stored": (
                cache_stored
            ),
        }

        if cache_key is not None:
            metadata[
                "cache_key_sha256"
            ] = cache_key

        return response.model_copy(
            update={
                "provider_id": (
                    self.provider_id
                ),
                "runtime_id": (
                    self.runtime_id
                ),
                "metadata": metadata,
            }
        )

    def _cache_key(
        self,
        request: ProviderInferenceRequest,
    ) -> str:
        """Hash exact deterministic request semantics excluding correlation ID."""
        payload = {
            "provider_id": (
                self._provider.provider_id
            ),
            "runtime_id": (
                self._provider.runtime_id
            ),
            "request": request.model_dump(
                mode="json",
                exclude={
                    "request_id"
                },
            ),
        }

        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
        )

        return sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _validate_request_identity(
        *,
        request: ProviderInferenceRequest,
        response: ProviderInferenceResponse,
    ) -> None:
        """Reject provider responses that cannot be correlated exactly."""
        if (
            response.request_id
            != request.request_id
        ):
            raise RuntimeError(
                "Provider response request_id "
                "does not match the cache request."
            )
