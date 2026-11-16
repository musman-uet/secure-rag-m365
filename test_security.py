"""Security trimming: a user must never retrieve, be answered from, or learn
about a document they cannot already open."""

import json

import pytest

from securerag.models import User
from securerag.pipeline import NOT_FOUND_MESSAGE
from tests.conftest import ROOT

QUESTIONS = [q["question"] for q in json.loads((ROOT / "eval" / "golden_set.json").read_text())["questions"]]


@pytest.mark.parametrize("mode", ["keyword", "vector", "hybrid"])
def test_no_user_ever_retrieves_a_document_they_cannot_access(pipeline, directory, acl, mode):
    checked = 0
    for user in directory.all_users():
        for q in QUESTIONS:
            for hit in pipeline.retrieve(q, user, top_k=10, mode=mode):
                checked += 1
                assert set(user.principals) & acl[hit.chunk.doc_id], (user.user_id, hit.chunk.doc_id, q)
    assert checked > 500


def test_user_without_identity_gets_nothing(indexed):
    backend, embedder, _ = indexed
    assert backend.search("annual leave", embedder.embed(["annual leave"])[0], principals=[]) == []


def test_restricted_answer_is_identical_to_nonexistent_answer(pipeline, directory):
    sara = directory.get_user("sara")  # engineering, not leadership
    restricted = pipeline.ask("What is Project Falcon?", sara)
    nonexistent = pipeline.ask("What is Project Zebra?", sara)
    assert restricted.answer == nonexistent.answer == NOT_FOUND_MESSAGE
    assert restricted.status == nonexistent.status
    assert restricted.citations == [] and nonexistent.citations == []


def test_restricted_content_never_appears_in_any_answer(pipeline, directory, indexed):
    _, _, chunks = indexed
    zain = directory.get_user("zain")  # contractor: it-security + contractors only
    allowed = set(zain.principals)
    secret_bits = ["Project Falcon", "600000", "19.6", "15 minutes", "20 days of paid annual leave"]
    for q in QUESTIONS:
        a = pipeline.ask(q, zain)
        for c in a.citations:
            doc_chunks = [x for x in chunks if x.doc_id == c.doc_id]
            assert allowed & set(doc_chunks[0].allowed_principals)
        for s in secret_bits:
            assert s not in a.answer


def test_access_follows_group_membership(pipeline):
    q = "What is the maximum monthly salary for grade G3?"
    hr = User("new-hr-person", ["hr-team"])
    finance = User("new-finance-person", ["finance-team"])
    assert pipeline.ask(q, hr).status.value == "answered"
    assert pipeline.ask(q, finance).status.value == "insufficient_evidence"


def test_access_can_be_granted_to_an_individual_user(indexed):
    from securerag.models import Chunk
    from securerag.search.local_backend import LocalSearchBackend

    b = LocalSearchBackend()
    b.index([Chunk("c1", "d1", "Secret plan", "S", "secret plan details", "u", "2025-01-01", "md",
                   ["user:alice@contoso.example"])])
    assert b.search("secret plan", None, ["user:alice@contoso.example"], mode="keyword")
    assert not b.search("secret plan", None, ["user:bob@contoso.example", "all-staff"], mode="keyword")
