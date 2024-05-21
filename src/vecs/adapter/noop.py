"""
The `vecs.experimental.adapter.noop` module provides a default no-op (no operation) adapter
that passes the inputs through without any modification. This can be useful when no specific
adapter processing is required.

All public classes, enums, and functions are re-exported by `vecs.adapters` module.
"""

from typing import Any, Generator, Iterable, Optional, Tuple

from .base import AdapterContext, AdapterStep
import uuid

class NoOp(AdapterStep):
    """
    NoOp is a no-operation AdapterStep. It is a default adapter that passes through
    the input records without any modifications.
    """

    def __init__(self, dimension: int):
        """
        Initializes the NoOp adapter with a dimension.

        Args:
            dimension (int): The dimension of the input vectors.
        """
        self._dimension = dimension

    @property
    def exported_dimension(self) -> Optional[int]:
        """
        Returns the dimension of the adapter.

        Returns:
            int: The dimension of the input vectors.
        """
        return self._dimension

    def __call__(
        self,
        records: Iterable[Tuple[Any, Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]],
        adapter_context: AdapterContext,
    ) -> Generator[Tuple[Any, int, int, int, int, int], None, None]:
        """
        Yields the input records without any modification.

        Args:
            records: Iterable of tuples each containing an id, a media, an optional dict and an optional str.
            adapter_context: Context of the adapter.

        Yields:
            Tuple[Any, int, int, int, int, int]: The input record.
        """
        for vector, document_instance_id, begin_offset_byte, chunk_bytes, offset_began, memento_membership in records:
            yield (vector, document_instance_id, begin_offset_byte, chunk_bytes, offset_began, memento_membership)
