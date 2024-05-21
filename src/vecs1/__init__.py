from vecs1 import exc
from vecs1.client import Client
from vecs1.collection import (
    Collection,
    IndexArgsHNSW,
    IndexArgsIVFFlat,
    IndexMeasure,
    IndexMethod,
)

__project__ = "vecs1"
__version__ = "0.4.2"


__all__ = [
    "IndexArgsIVFFlat",
    "IndexArgsHNSW",
    "IndexMethod",
    "IndexMeasure",
    "Collection",
    "Client",
    "exc",
]


def create_client(connection_string: str) -> Client:
    """Creates a client from a Postgres connection string"""
    return Client(connection_string)
