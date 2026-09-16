"""Provider-neutral embedding interfaces."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EmbeddingEncoder(Protocol):
    """Encode text into dense numeric vectors."""

    @property
    def implementation_id(self) -> str:
        """Return a stable identifier for the embedding implementation."""
        ...

    def encode(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float64]:
        """Encode texts into a two-dimensional embedding matrix."""
        ...
