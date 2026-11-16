"""Azure AI Search backend - verified offline against the real SDK models."""

from types import SimpleNamespace

import pytest

from securerag.models import Chunk
from securerag.search.azure_backend import (AzureSearchBackend, UnsafePrincipalError, build_index,
                                            build_search_kwargs, build_security_filter)


def test_security_filter_format():
    assert build_security_filter(["ayesha", "all-staff"]) == \
        "allowed_principals/any(p: search.in(p, 'ayesha|all-staff', '|'))"


@pytest.mark.parametrize("bad", ["x') or true or ('", "a|b", "a,b", "o'brien", "", "a b"])
def test_filter_rejects_injection_and_delimiters(bad):
    with pytest.raises(UnsafePrincipalError):
        build_security_filter(["all-staff", bad])


def test_filter_requires_an_identity():
    with pytest.raises(UnsafePrincipalError):
        build_security_filter([])


def test_entra_object_ids_and_upns_are_accepted():
    f = build_security_filter(["5f1c2d3e-1111-2222-3333-444455556666", "alice@contoso.example"])
    assert "5f1c2d3e-1111-2222-3333-444455556666|alice@contoso.example" in f


def test_index_schema():
    d = build_index("chunks", 1536).as_dict()
    fields = {f["name"]: f for f in d["fields"]}
    acl = fields["allowed_principals"]
    assert acl["type"] == "Collection(Edm.String)" and acl["filterable"] and not acl["retrievable"]
    assert fields["id"]["key"]
    assert fields["content_vector"]["dimensions"] == 1536
    # Keys below are the Azure AI Search REST API names produced by the SDK.
    assert fields["content_vector"]["vectorSearchProfile"] == "default-vector-profile"
    assert d["vectorSearch"]["profiles"][0]["algorithm"] == "default-hnsw"
    assert d["vectorSearch"]["algorithms"][0]["kind"] == "hnsw"
    sem = d["semantic"]["configurations"][0]
    assert sem["name"] == "default-semantic"
    assert sem["prioritizedFields"]["titleField"]["fieldName"] == "title"


def test_hybrid_semantic_query_arguments():
    kw = build_search_kwargs("annual leave", [0.0] * 4, ["ayesha", "hr-team"], 5, "hybrid", True)
    assert kw["search_text"] == "annual leave"
    assert kw["vector_filter_mode"] == "preFilter"  # vector leg trimmed BEFORE kNN
    assert kw["query_type"] == "semantic" and kw["semantic_configuration_name"] == "default-semantic"
    assert "search.in(p, 'ayesha|hr-team', '|')" in kw["filter"]
    vq = kw["vector_queries"][0]
    assert vq.fields == "content_vector" and vq.k_nearest_neighbors == 50
    assert "allowed_principals" not in kw["select"]


def test_vector_mode_requires_vector():
    with pytest.raises(ValueError):
        build_search_kwargs("q", None, ["g"], 5, "vector", False)


class FakeSearchClient:
    def __init__(self):
        self.calls, self.uploaded = [], []

    def search(self, **kw):
        self.calls.append(kw)
        return [{"id": "c1", "doc_id": "handbook", "title": "Employee Handbook", "section": "Annual Leave",
                 "content": "20 days", "source_url": "https://x", "last_modified": "2025-01-10",
                 "file_type": "docx", "page": None, "@search.score": 0.03, "@search.reranker_score": 3.2}]

    def merge_or_upload_documents(self, documents):
        self.uploaded.extend(documents)
        return [SimpleNamespace(key=d["id"], succeeded=True) for d in documents]


def test_search_maps_results_and_reranker_score():
    client = FakeSearchClient()
    b = AzureSearchBackend("e", "k", "idx", 4, search_client=client, index_client=object())
    [hit] = b.search("annual leave", [0.1] * 4, ["ayesha", "all-staff"])
    assert hit.chunk.doc_id == "handbook" and hit.relevance == pytest.approx(0.8)
    assert client.calls[0]["filter"].startswith("allowed_principals/any(")
    assert b.provides_semantic_relevance


def test_upload_documents_shape():
    client = FakeSearchClient()
    b = AzureSearchBackend("e", "k", "idx", 2, search_client=client, index_client=object())
    b.index([Chunk("c1", "d1", "T", "S", "text", "u", "2025-01-01", "md", ["g1"], vector=[0.1, 0.2])])
    doc = client.uploaded[0]
    assert doc["id"] == "c1" and doc["content_vector"] == [0.1, 0.2] and doc["allowed_principals"] == ["g1"]
    assert "vector" not in doc and "page" not in doc


def test_upload_refuses_chunk_without_vector():
    b = AzureSearchBackend("e", "k", "idx", 2, search_client=FakeSearchClient(), index_client=object())
    with pytest.raises(ValueError):
        b.index([Chunk("c1", "d1", "T", "S", "text", "u", "2025-01-01", "md", ["g1"])])
