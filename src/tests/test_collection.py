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
            ix,
            vec,
            1,
            0,
            100,
            1714039150,
            1,
            1,
            "4f67a14f-dfd5-4382-b60b-e520888d15d8"
        )
        for ix, vec in enumerate(np.random.random((n_records, dim)))
    ]

    # insert works
    movies.upsert(records)
    assert len(movies) == n_records

    # upserting overwrites
    new_record = (1, np.zeros(384), 1, 1, 100, 1714039150, 1, 1, "4f67a14f-dfd5-4382-b60b-e520888d15d8")
    movies.upsert([new_record])
    db_record = movies[1]
    db_record[0] == new_record[0]
    db_record[1] == new_record[1]
    db_record[2] == new_record[2]


def test_fetch(client: vecs.Client) -> None:
    n_records = 100
    dim = 384

    movies = client.get_or_create_collection(name="ping", dimension=dim)

    records = [
        (
            ix,
            vec,
            1,
            0,
            100,
            1714039150,
            1,
            1,
            "4f67a14f-dfd5-4382-b60b-e520888d15d8"
        )
        for ix, vec in enumerate(np.random.random((n_records, dim)))
    ]

    # insert works
    movies.upsert(records)

    # test basic usage
    fetch_ids = [1, 12, 99]
    res = movies.fetch(ids=fetch_ids)
    assert len(res) == 3
    ids = set([x[0] for x in res])
    assert all([x in ids for x in fetch_ids])

    # test one of the keys does not exist not an error
    fetch_ids = [1, 12, 133]
    res = movies.fetch(ids=fetch_ids)
    assert len(res) == 2

    # bad input
    with pytest.raises(vecs.exc.ArgError):
        movies.fetch(ids="should_be_a_list")


def test_delete(client: vecs.Client) -> None:
    n_records = 100
    dim = 384

    movies = client.get_or_create_collection(name="ping", dimension=dim)

    records = [
        (
            ix,
            vec,
            1,
            0,
            100,
            1714039150,
            1,
            1,
            "4f67a14f-dfd5-4382-b60b-e520888d15d8"
        )
        for (ix, vec), genre in zip(
            enumerate(np.random.random((n_records, dim))),
            itertools.cycle(["action", "rom-com", "drama"]),
        )
    ]

    # insert works
    movies.upsert(records)

    # delete by IDs.
    delete_ids = [1, 12, 99]
    movies.delete(ids=delete_ids)
    assert len(movies) == n_records - len(delete_ids)

    # insert works
    movies.upsert(records)

    # bad input
    with pytest.raises(vecs.exc.ArgError):
        movies.delete(ids="should_be_a_list")

    # bad input: neither ids nor filters provided.
    with pytest.raises(vecs.exc.ArgError):
        movies.delete()


def test_repr(client: vecs.Client) -> None:
    movies = client.get_or_create_collection(name="movies", dimension=99)
    assert repr(movies) == 'vecs.Collection(name="movies", dimension=99)'


def test_getitem(client: vecs.Client) -> None:
    movies = client.get_or_create_collection(name="movies", dimension=3)
    movies.upsert(records=[(1, [1, 2, 3], 1, 1, 100, 1714039150, 1, 1, "4f67a14f-dfd5-4382-b60b-e520888d15d8")])

    assert movies[1] is not None
    assert len(movies[1]) == 9

    with pytest.raises(KeyError):
        assert movies[2] is not None

    with pytest.raises(vecs.exc.ArgError):
        movies[["only strings work not lists"]]


@pytest.mark.filterwarnings("ignore:Query does")
def test_query(client: vecs.Client) -> None:
    n_records = 100
    dim = 64

    bar = client.get_or_create_collection(name="bar", dimension=dim)

    records = [
        (
            ix,
            vec,
            1,
            0,
            100,
            1714039150,
            1,
            1,
            "4f67a14f-dfd5-4382-b60b-e520888d15d8"
        )
        for ix, vec in enumerate(np.random.random((n_records, dim)))
    ]

    bar.upsert(records)

    _, query_vec, query_document_content_id, query_begin_offset_byte, query_chunk_bytes, query_offset_began_hhmm1970, query_memento_membership, _, _ = bar[5]

    top_k = 7

    res = bar.query(
        data=query_vec,
        limit=top_k,
        # filters=None,
        measure="cosine_distance",
        include_value=False,
        include_metadata=False,
    )

    # correct number of results
    assert len(res) == top_k
    # most similar to self
    assert res[0] == '5'

    with pytest.raises(vecs.exc.ArgError):
        res = bar.query(
            data=query_vec,
            limit=1001,
        )

    with pytest.raises(vecs.exc.ArgError):
        res = bar.query(
            data=query_vec,
            probes=0,
        )

    with pytest.raises(vecs.exc.ArgError):
        res = bar.query(
            data=query_vec,
            probes=-1,
        )

    with pytest.raises(vecs.exc.ArgError):
        res = bar.query(
            data=query_vec,
            probes="a",  # type: ignore
        )

    with pytest.raises(vecs.exc.ArgError):
        res = bar.query(data=query_vec, limit=top_k, measure="invalid")

    # skip_adapter has no effect (no adapter present)
    res = bar.query(data=query_vec, limit=top_k, skip_adapter=True)
    assert len(res) == top_k

    # include_value
    res = bar.query(
        data=query_vec,
        limit=top_k,
        measure="cosine_distance",
        include_value=True,
    )
    assert len(res[0]) == 2
    assert res[0][0] == 5
    assert pytest.approx(res[0][1]) == 0

    # include_metadata
    res = bar.query(
        data=query_vec,
        limit=top_k,
        measure="cosine_distance",
        include_metadata=True,
    )
    assert len(res[0]) == 8
    assert res[0][0] == 5
    assert res[0][1] == query_document_content_id
    assert res[0][2] == query_begin_offset_byte
    assert res[0][3] == query_chunk_bytes

    res = bar.query(
        data=query_vec,
        limit=top_k,
        measure="cosine_distance",
    )
    assert len(res[0]) == 1
    assert res[0][0] == '5'

    # include_value, include_metadata
    res = bar.query(
        data=query_vec,
        limit=top_k,
        measure="cosine_distance",
        include_metadata=True,
        include_value=True,
    )
    assert len(res[0]) == 9
    assert res[0][0] == 5
    assert pytest.approx(res[0][1]) == 0
    assert res[0][2] == query_document_content_id
    assert res[0][3] == query_begin_offset_byte
    assert res[0][4] == query_chunk_bytes
    assert res[0][5] == query_offset_began_hhmm1970
    assert res[0][6] == query_memento_membership

    # test for different numbers of probes
    assert len(bar.query(data=query_vec, limit=top_k, probes=10)) == top_k

    assert len(bar.query(data=query_vec, limit=top_k, probes=5)) == top_k

    assert len(bar.query(data=query_vec, limit=top_k, probes=1)) == top_k

    assert len(bar.query(data=query_vec, limit=top_k, probes=999)) == top_k


def test_access_index(client: vecs.Client) -> None:
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)
    assert bar.index is None


def test_create_index(client: vecs.Client) -> None:
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)

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
    bar.upsert([(1, [1, 2, 3, 4], 1, 1, 100, 1714039150, 1, 1, "4f67a14f-dfd5-4382-b60b-e520888d15d8")])

    bar.create_index(method="ivfflat")  # type: ignore
    results = bar.query(data=[1, 2, 3, 4], limit=1, probes=50)
    assert len(results) == 1

    bar.create_index(method=IndexMethod.ivfflat, replace=True)  # type: ignore
    results = bar.query(
        data=[1, 2, 3, 4],
        limit=1,
    )
    assert len(results) == 1


def test_hnsw(client: vecs.Client) -> None:
    dim = 4
    bar = client.get_or_create_collection(name="bar", dimension=dim)
    bar.upsert([(1, [1, 2, 3, 4], 1, 1, 100, 1714039150, 1, 1, "4f67a14f-dfd5-4382-b60b-e520888d15d8")])

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
    bar.upsert([(1, [1, 2, 3, 4], 1, 1, 100, 1714039150, 1, 1, "4f67a14f-dfd5-4382-b60b-e520888d15d8")])

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
    bar.upsert([(1, [1, 2, 3, 4], 1, 1, 100, 1714039150, 1, 1, "4f67a14f-dfd5-4382-b60b-e520888d15d8")])
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
    bar.upsert([(1, [1, 2, 3, 4], 1, 1, 100, 1714039150, 1, 1, "4f67a14f-dfd5-4382-b60b-e520888d15d8")])
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
    bar.upsert([(1, [1, 2, 3, 4], 1, 1, 100, 1714039150, 1, 1, "4f67a14f-dfd5-4382-b60b-e520888d15d8")])
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
    bar.upsert([(1, [1, 2, 3, 4], 1, 1, 100, 1714039150, 1, 1, "4f67a14f-dfd5-4382-b60b-e520888d15d8")])
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
    bar.upsert([(1, [1, 2, 3, 4], 1, 1, 100, 1714039150, 1, 1, "4f67a14f-dfd5-4382-b60b-e520888d15d8")])
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
