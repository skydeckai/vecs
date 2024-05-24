from vecs_new import exc
from vecs_new.client import Client
from vecs_new.collection import (
    Collection,
    IndexArgsHNSW,
    IndexArgsIVFFlat,
    IndexMeasure,
    IndexMethod,
)

__project__ = "vecs_new"
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
