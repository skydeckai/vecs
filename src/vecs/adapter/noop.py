"""
The `vecs.experimental.adapter.noop` module provides a default no-op (no operation) adapter
that passes the inputs through without any modification. This can be useful when no specific
adapter processing is required.

All public classes, enums, and functions are re-exported by `vecs.adapters` module.
"""

from typing import Any, Dict, Generator, Iterable, Optional, Tuple

from .base import AdapterContext, AdapterStep


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
        records: Iterable[
            Tuple[
                str,
                Any,
                Optional[Dict],
                Optional[str],
                Optional[int],
                Optional[int],
                Optional[int],
                Optional[int],
            ]
        ],
        adapter_context: AdapterContext,
    ) -> Generator[Tuple[str, Any, Dict, str, int, int, int, int], None, None]:
        """
        Yields the input records without any modification.

        Args:
            records: Iterable of tuples each containing an id, a media, an optional dict and an optional str.
            adapter_context: Context of the adapter.

        Yields:
            Tuple[str, Any, Dict, str, int, int, int]: The input record.
        """
        NUM_REQUIRED_FIELDS = 6
        NUM_OPTIONAL_EXTRA_FIELDS = 2  # memento_membership, app_id
        MAX_NUM_FIELDS = NUM_REQUIRED_FIELDS + NUM_OPTIONAL_EXTRA_FIELDS
        for record in records:
            if len(record) < MAX_NUM_FIELDS:
                assert len(record) >= NUM_REQUIRED_FIELDS
                record = list(record) + [None] * (MAX_NUM_FIELDS - len(record))
            (
                id,
                media,
                metadata,
                text,
                doc_instance_id,
                order,
                memento_membership,
                app_id,
            ) = record
            yield (
                id,
                media,
                metadata or {},
                text or None,
                doc_instance_id,
                order,
                memento_membership,
                app_id,
            )
