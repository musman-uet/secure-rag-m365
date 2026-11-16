import json

from securerag.evidence import assess
from securerag.generation import GeneratedAnswer, extract_refs, parse_llm_json
from securerag.models import AnswerStatus, Chunk, SearchHit
from securerag.pipeline import RagPipeline


def _hit(doc_id, title, content, date="2025-01-01", relevance=None):
    return SearchHit(Chunk(doc_id + "-c", doc_id, title, "S", content, "u", date, "pdf", ["g"]),
                     1.0, relevance)


# --------------------------------------------------------------- retrieval

def test_hybrid_ranks_the_right_document_first(pipeline, directory):
    hits = pipeline.retrieve("How quickly must a Sev1 incident be acknowledged?", directory.get_user("sara"))
    assert hits[0].chunk.doc_id == "runbook"


def test_keyword_mode_needs_no_vector(pipeline, directory):
    hits = pipeline.retrieve("multi-factor authentication VPN", directory.get_user("zain"), mode="keyword")
    assert hits and hits[0].chunk.doc_id == "it-security"


def test_hybrid_recovers_paraphrase_that_keyword_misses(pipeline, directory):
    user = directory.get_user("ayesha")
    q = "How much maternity leave is offered?"
    hybrid_docs = [h.chunk.doc_id for h in pipeline.retrieve(q, user, mode="hybrid")]
    assert "handbook" in hybrid_docs


# ---------------------------------------------------------------- evidence

def test_no_hits_is_insufficient():
    assert not assess("anything", []).sufficient


def test_unrelated_hits_are_insufficient():
    hits = [_hit("a", "Contractor guide", "Contractors receive a company account."),
            _hit("b", "IT policy", "Company laptops must be encrypted.")]
    r = assess("What is the company's stock price?", hits)
    assert not r.sufficient and r.reasons


def test_conflict_between_documents_is_flagged_with_dates():
    hits = [_hit("t23", "Travel Policy 2023", "The daily meal allowance for domestic travel is PKR 3,000 per day.", "2023-02-01"),
            _hit("t25", "Travel Policy 2025", "The daily meal allowance for domestic travel is PKR 4,500 per day.", "2025-03-15")]
    r = assess("What is the daily meal allowance for domestic travel?", hits)
    assert r.sufficient and len(r.conflicts) == 1
    assert "PKR 3,000" in r.conflicts[0] and "PKR 4,500" in r.conflicts[0] and "2025-03-15" in r.conflicts[0]


def test_same_figures_in_two_documents_is_not_a_conflict():
    text = "The daily meal allowance for domestic travel is PKR 4,500 per day."
    r = assess("What is the daily meal allowance for domestic travel?",
               [_hit("a", "A", text), _hit("b", "B", text)])
    assert r.sufficient and not r.conflicts


def test_semantic_relevance_gate_is_used_when_available():
    paraphrase = [_hit("h", "Handbook", "Birthing parents receive 16 weeks of paid parental leave.", relevance=0.8)]
    assert assess("How much maternity leave is offered?", paraphrase, semantic=True).sufficient
    weak = [_hit("h", "Handbook", "Birthing parents receive 16 weeks of paid parental leave.", relevance=0.2)]
    assert not assess("How much maternity leave is offered?", weak, semantic=True).sufficient


# --------------------------------------------------------------- answering

def test_answer_has_citations_with_section_page_date_and_link(pipeline, directory):
    a = pipeline.ask("What is the mileage reimbursement per km?", directory.get_user("omar"))
    assert a.status == AnswerStatus.ANSWERED and a.ai_generated
    c = a.citations[0]
    assert c.doc_id == "travel-2025" and c.section == "Mileage" and c.page == 2
    assert c.last_modified == "2025-03-15" and c.source_url.startswith("https://")
    assert "[1]" in a.answer and "65" in a.answer


def test_conflicting_policies_return_both_versions(pipeline, directory):
    a = pipeline.ask("What is the daily meal allowance for domestic travel?", directory.get_user("bilal"))
    assert a.status == AnswerStatus.CONFLICTING_EVIDENCE
    assert {c.doc_id for c in a.citations} == {"travel-2023", "travel-2025"}
    assert any("disagree" in n for n in a.notes)


def test_citation_numbers_are_sequential(pipeline, directory):
    a = pipeline.ask("How quickly must the on-call engineer acknowledge a Sev1 incident?",
                     directory.get_user("sara"))
    assert [c.ref for c in a.citations] == list(range(1, len(a.citations) + 1))


class _FakeGenerator:
    def __init__(self, result):
        self.result = result

    def generate(self, question, hits):
        return self.result


def test_uncited_llm_answer_is_never_shown(indexed, directory):
    backend, embedder, _ = indexed
    p = RagPipeline(backend, embedder, _FakeGenerator(GeneratedAnswer("Everyone gets 99 days.", [])))
    a = p.ask("How many days of annual leave do employees get?", directory.get_user("ayesha"))
    assert a.status == AnswerStatus.INSUFFICIENT_EVIDENCE and "99" not in a.answer


def test_refs_outside_the_source_list_are_dropped():
    assert extract_refs("A [1]. B [7]. C [2][1].", n_sources=3) == [1, 2]


def test_llm_json_parsing():
    g = parse_llm_json(json.dumps({"status": "answered", "answer": "20 days [1]."}), 5)
    assert g.refs == [1] and not g.insufficient and not g.conflict
    assert parse_llm_json("not json", 5).insufficient
    assert parse_llm_json(json.dumps({"status": "insufficient_evidence", "answer": ""}), 5).insufficient
    assert parse_llm_json(json.dumps({"status": "conflicting_evidence", "answer": "A [1] B [2]"}), 5).conflict


def test_azure_openai_generator_with_fake_client():
    from types import SimpleNamespace

    from securerag.generation import AzureOpenAIGenerator

    captured = {}

    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            msg = SimpleNamespace(content=json.dumps({"status": "answered", "answer": "PKR 65 per km [1]."}))
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    gen = AzureOpenAIGenerator("e", "k", "gpt-4o-mini", client=client)
    out = gen.generate("mileage?", [_hit("t25", "Travel Policy 2025", "PKR 65 per km.")])
    assert out.refs == [1]
    assert captured["temperature"] == 0 and captured["response_format"] == {"type": "json_object"}
    assert "[1] Travel Policy 2025" in captured["messages"][1]["content"]


def test_every_question_is_audited(pipeline, directory, tmp_path):
    pipeline.ask("How long must passwords be?", directory.get_user("zain"))
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    entry = json.loads(lines[-1])
    assert entry["user_id"] == "zain" and entry["status"] == "answered"
    assert entry["cited_doc_ids"] == ["it-security"] and "timestamp" in entry
    assert "14 characters" not in lines[-1]  # document content is not logged
