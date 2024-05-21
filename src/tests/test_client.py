import pytest
import sqlalchemy

import vecs1

def test_create_collection(client: vecs1.Client) -> None:
    with pytest.warns(DeprecationWarning):
        client.create_collection(name="docs", dimension=384)

        with pytest.raises(sqlalchemy.exc.InvalidRequestError):
            client.create_collection(name="docs", dimension=384)
