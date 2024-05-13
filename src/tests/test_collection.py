import itertools

import numpy as np
import pytest

import vecs
from vecs import IndexArgsHNSW, IndexArgsIVFFlat, IndexMethod
from vecs.exc import ArgError


def test_upsert(client: vecs.Client) -> None:
    n_records = 100
    dim = 384

    movies = client.get_or_create_collection(name="ping", dimension=dim)

    # collection initially empty
    assert len(movies) == 0

    records = [
        (
            vec,
            ix,
            2,
            100,
            1714039150,
            1,
        )
        for ix, vec in enumerate(np.random.random((n_records, dim)))
    ]

    # insert works
    movies.upsert(records)
    assert len(movies) == n_records

def test_create_index(client: vecs.Client) -> None:
    dim = 4
    bar = client.create_collection(name="create-index-collection", dimension=dim)

    bar.create_index()

    assert bar.index is not None

    with pytest.raises(vecs.exc.ArgError):
        bar.create_index(replace=False)

    with pytest.raises(vecs.exc.ArgError):
        bar.create_index(method="does not exist")

    with pytest.raises(vecs.exc.ArgError):
        bar.create_index(measure="does not exist")

    bar.query(
        data=[1, 2, 3, 4],
        limit=1,
        measure="cosine_distance",
    )


def test_ivfflat(client: vecs.Client) -> None:
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)
    bar.upsert([([1, 2, 3, 4], 1, 1, 100, 1714039150, 1)])

    bar.create_index(method="ivfflat")
    results = bar.query(data=[1, 2, 3, 4], limit=1, probes=50)
    assert len(results) == 1

    bar.create_index(method=IndexMethod.ivfflat, replace=True)
    results = bar.query(
        data=[1, 2, 3, 4],
        limit=1,
    )
    assert len(results) == 1


def test_hnsw(client: vecs.Client) -> None:
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)
    bar.upsert([([1, 2, 3, 4], 1, 1, 100, 1714039150, 1)])

    bar.create_index(method="hnsw")  # type: ignore
    results = bar.query(
        data=[1, 2, 3, 4],
        limit=1,
    )
    assert len(results) == 1

    bar.create_index(method=IndexMethod.hnsw, replace=True)  # type: ignore
    results = bar.query(data=[1, 2, 3, 4], limit=1, ef_search=50)
    assert len(results) == 1


def test_index_build_args(client: vecs.Client) -> None:
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)
    bar.upsert([([1, 2, 3, 4], 1, 1, 100, 1714039150, 1)])

    # Test that default value for nlists is used in absence of index build args
    bar.create_index(method="ivfflat")
    [nlists] = [i for i in bar.index.split("_") if i.startswith("nl")]
    assert int(nlists.strip("nl")) == 30

    # Test nlists is honored when supplied
    bar.create_index(
        method=IndexMethod.ivfflat,
        index_arguments=IndexArgsIVFFlat(n_lists=123),
        replace=True,
    )
    [nlists] = [i for i in bar.index.split("_") if i.startswith("nl")]
    assert int(nlists.strip("nl")) == 123

    # Test that default values for m and ef_construction are used in absence of
    # index build args
    bar.create_index(method="hnsw", replace=True)
    [m] = [i for i in bar.index.split("_") if i.startswith("m")]
    [ef_construction] = [i for i in bar.index.split("_") if i.startswith("efc")]
    assert int(m.strip("m")) == 16
    assert int(ef_construction.strip("efc")) == 64

    # Test m and ef_construction is honored when supplied
    bar.create_index(
        method="hnsw",
        index_arguments=IndexArgsHNSW(m=8, ef_construction=123),
        replace=True,
    )
    [m] = [i for i in bar.index.split("_") if i.startswith("m")]
    [ef_construction] = [i for i in bar.index.split("_") if i.startswith("efc")]
    assert int(m.strip("m")) == 8
    assert int(ef_construction.strip("efc")) == 123

    # Test m is honored and ef_construction is default when _only_ m is supplied
    bar.create_index(method="hnsw", index_arguments=IndexArgsHNSW(m=8), replace=True)
    [m] = [i for i in bar.index.split("_") if i.startswith("m")]
    [ef_construction] = [i for i in bar.index.split("_") if i.startswith("efc")]
    assert int(m.strip("m")) == 8
    assert int(ef_construction.strip("efc")) == 64

    # Test m is default and ef_construction is honoured when _only_
    # ef_construction is supplied
    bar.create_index(
        method="hnsw", index_arguments=IndexArgsHNSW(ef_construction=123), replace=True
    )
    [m] = [i for i in bar.index.split("_") if i.startswith("m")]
    [ef_construction] = [i for i in bar.index.split("_") if i.startswith("efc")]
    assert int(m.strip("m")) == 16
    assert int(ef_construction.strip("efc")) == 123

    # Test that exception is raised when index build args don't match
    # the requested index type
    with pytest.raises(vecs.exc.ArgError):
        bar.create_index(
            method=IndexMethod.ivfflat, index_arguments=IndexArgsHNSW(), replace=True
        )
    with pytest.raises(vecs.exc.ArgError):
        bar.create_index(
            method=IndexMethod.hnsw,
            index_arguments=IndexArgsIVFFlat(n_lists=123),
            replace=True,
        )

    # Test that excpetion is raised index build args are supplied by the
    # IndexMethod.auto index is specified
    with pytest.raises(vecs.exc.ArgError):
        bar.create_index(
            method=IndexMethod.auto,
            index_arguments=IndexArgsIVFFlat(n_lists=123),
            replace=True,
        )
    with pytest.raises(vecs.exc.ArgError):
        bar.create_index(
            method=IndexMethod.auto,
            index_arguments=IndexArgsHNSW(),
            replace=True,
        )


def test_cosine_index_query(client: vecs.Client) -> None:
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)
    bar.upsert([([1, 2, 3, 4], 1, 1, 100, 1714039150, 1)])
    bar.create_index(measure=vecs.IndexMeasure.cosine_distance)
    results = bar.query(
        data=[1, 2, 3, 4],
        limit=1,
        measure="cosine_distance",
    )
    assert len(results) == 1


def test_l2_index_query(client: vecs.Client) -> None:
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)
    bar.upsert([([1, 2, 3, 4], 1, 1, 100, 1714039150, 1)])
    bar.create_index(measure=vecs.IndexMeasure.l2_distance)
    results = bar.query(
        data=[1, 2, 3, 4],
        limit=1,
        measure="l2_distance",
    )
    assert len(results) == 1


def test_max_inner_product_index_query(client: vecs.Client) -> None:
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)
    bar.upsert([([1, 2, 3, 4], 1, 1, 100, 1714039150, 1)])
    bar.create_index(measure=vecs.IndexMeasure.max_inner_product)
    results = bar.query(
        data=[1, 2, 3, 4],
        limit=1,
        measure="max_inner_product",
    )
    assert len(results) == 1


def test_mismatch_measure(client: vecs.Client) -> None:
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)
    bar.upsert([([1, 2, 3, 4], 1, 1, 100, 1714039150, 1)])
    bar.create_index(measure=vecs.IndexMeasure.max_inner_product)
    with pytest.warns(UserWarning):
        results = bar.query(
            data=[1, 2, 3, 4],
            limit=1,
            # wrong measure
            measure="cosine_distance",
        )
    assert len(results) == 1


def test_is_indexed_for_measure(client: vecs.Client) -> None:
    bar = client.get_or_create_collection(name="bar", dimension=4)

    bar.create_index(measure=vecs.IndexMeasure.max_inner_product)
    assert not bar.is_indexed_for_measure("invalid")  # type: ignore
    assert bar.is_indexed_for_measure(vecs.IndexMeasure.max_inner_product)
    assert not bar.is_indexed_for_measure(vecs.IndexMeasure.cosine_distance)

    bar.create_index(measure=vecs.IndexMeasure.cosine_distance, replace=True)
    assert bar.is_indexed_for_measure(vecs.IndexMeasure.cosine_distance)


def test_failover_ivfflat(client: vecs.Client) -> None:
    """Test that index fails over to ivfflat on 0.4.0
    This is already covered by CI's test matrix but it is convenient for faster feedback
    to include it when running on the latest version of pgvector
    """
    client.vector_version = "0.4.1"
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)
    bar.upsert([([1, 2, 3, 4], 1, 1, 100, 1714039150, 1)])
    # this executes an otherwise uncovered line of code that selects ivfflat when mode is 'auto'
    # and hnsw is unavailable
    bar.create_index(method=IndexMethod.auto)


def test_hnsw_unavailable_error(client: vecs.Client) -> None:
    """Test that index fails over to ivfflat on 0.4.0
    This is already covered by CI's test matrix but it is convenient for faster feedback
    to include it when running on the latest version of pgvector
    """
    client.vector_version = "0.4.1"
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)
    with pytest.raises(ArgError):
        bar.create_index(method=IndexMethod.hnsw)
