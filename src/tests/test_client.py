import pytest
import sqlalchemy

import vecs_new

def test_create_collection(client: vecs_new.Client) -> None:
    with pytest.warns(DeprecationWarning):
        client.create_collection(name="docs", dimension=384)

        with pytest.raises(vecs_new.exc.CollectionAlreadyExists):
            client.create_collection(name="docs", dimension=384)
