import pytest

from app.services.rrf_service import reciprocal_rank_fusion


def test_rrf_prioritizes_shared_results():
    semantic_results = [
        {"chunk_id": "a", "score": 0.95},
        {"chunk_id": "b", "score": 0.85},
    ]

    keyword_results = [
        {"chunk_id": "b", "score": 12.5},
        {"chunk_id": "c", "score": 10.0},
    ]

    results = reciprocal_rank_fusion(
        [semantic_results, keyword_results],
        top_k=3,
    )

    assert [result["chunk_id"] for result in results] == ["b", "a", "c"]


def test_rrf_deduplicates_within_result_list():
    results = reciprocal_rank_fusion(
        [[
            {"chunk_id": "a", "score": 0.9},
            {"chunk_id": "a", "score": 0.8},
        ]],
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"


def test_rrf_empty_results():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_invalid_parameters():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=0)

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], top_k=0)