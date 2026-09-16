"""Registry for discoverable VAIT workflow transformations."""

from vait.transformations.models import (
    Transformation,
    TransformationDescriptor,
)


class TransformationRegistry:
    """In-memory registry of versioned workflow transformations."""

    def __init__(self) -> None:
        """Create an empty transformation registry."""
        self._transformations: dict[str, Transformation] = {}

    def register(
        self,
        transformation: Transformation,
    ) -> None:
        """Register one transformation by canonical identifier."""
        canonical_id = transformation.descriptor.canonical_id

        if canonical_id in self._transformations:
            raise ValueError(
                f"Transformation '{canonical_id}' is already registered."
            )

        self._transformations[canonical_id] = transformation

    def get(
        self,
        transformation_id: str,
        version: str,
    ) -> Transformation:
        """Return one registered transformation."""
        canonical_id = f"{transformation_id}@{version}"

        try:
            return self._transformations[canonical_id]
        except KeyError as exc:
            raise KeyError(
                f"Transformation '{canonical_id}' is not registered."
            ) from exc

    def list_descriptors(
        self,
    ) -> tuple[TransformationDescriptor, ...]:
        """Return registered descriptors in deterministic order."""
        return tuple(
            self._transformations[canonical_id].descriptor
            for canonical_id in sorted(self._transformations)
        )

    def __len__(self) -> int:
        """Return the number of registered transformations."""
        return len(self._transformations)
