import pytest
import sqlalchemy

import vecs


def test_create_collection(client: vecs.Client) -> None:
    with pytest.warns(DeprecationWarning):
        client.create_collection(name="docs", dimension=384)

        with pytest.raises(vecs.exc.CollectionAlreadyExists):
            client.create_collection(name="docs", dimension=384)
