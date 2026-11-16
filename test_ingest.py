import json
import re

import pytest
from pypdf import PdfWriter

from securerag.ingest.chunker import chunk_document
from securerag.ingest.loader import load_documents
from securerag.ingest.parsers import parse_file
from securerag.models import Document, Section
from tests.conftest import DATA

DOCS = DATA / "docs"


def test_docx_sections_follow_headings_and_tables_are_kept():
    sections = parse_file(DOCS / "incident_response_runbook.docx")
    headings = [s.heading for s in sections]
    assert {"Severity Levels", "Response Times", "Escalation", "Postmortems"} <= set(headings)
    table = next(s for s in sections if s.heading == "Table 1")
    assert "Sev1 | 15 minutes | 30 minutes" in table.text


def test_pdf_sections_have_page_numbers_and_headings():
    sections = parse_file(DOCS / "travel_policy_2025.pdf")
    meal = next(s for s in sections if s.heading == "Daily meal allowance")
    assert meal.page == 2
    assert "PKR 4,500 per day" in meal.text


def test_scanned_pdf_page_is_flagged_for_ocr(tmp_path):
    path = tmp_path / "scan.pdf"
    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    with path.open("wb") as f:
        w.write(f)
    [section] = parse_file(path)
    assert "needs OCR" in section.text and section.page == 1


def test_xlsx_rows_keep_header_names():
    sections = parse_file(DOCS / "salary_bands_2025.xlsx")
    bands = next(s for s in sections if s.heading == "Sheet Bands")
    assert "Grade: G3; Role level: Senior Engineer / Senior Analyst" in bands.text
    assert "Maximum monthly salary (PKR): 600000" in bands.text


def test_markdown_headings():
    headings = [s.heading for s in parse_file(DOCS / "it_security_policy.md")]
    assert headings == ["Passwords", "Multi-factor authentication", "Laptops", "Reporting"]


def test_unsupported_file_type(tmp_path):
    p = tmp_path / "x.pptx"
    p.write_bytes(b"")
    with pytest.raises(ValueError):
        parse_file(p)


def _doc(text: str, heading="Body") -> Document:
    return Document("d1", "Doc", "https://x", "2025-01-01", "txt", [Section(heading, text)], ["all-staff"])


def test_chunks_never_cross_sections_and_carry_metadata():
    doc = Document("d1", "Doc", "https://x", "2025-01-01", "txt",
                   [Section("A", "Alpha text here."), Section("B", "Beta text here.")], ["g1"])
    chunks = chunk_document(doc)
    assert [c.section for c in chunks] == ["A", "B"]
    assert all(c.allowed_principals == ["g1"] and c.last_modified == "2025-01-01" for c in chunks)
    assert chunks[0].content.startswith("Doc - A")


def test_long_section_split_with_overlap_and_safe_ids():
    sentences = " ".join(f"Sentence number {i} talks about leave policy." for i in range(100))
    chunks = chunk_document(_doc(sentences), max_words=50, overlap_words=10)
    assert len(chunks) > 5
    assert all(len(c.content.split()) <= 50 + 10 + 12 for c in chunks)
    # overlap: the start of each chunk repeats the end of the previous one
    first_body = chunks[0].content.split("\n", 1)[1].split()
    assert " ".join(first_body[-10:]) in chunks[1].content
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[A-Za-z0-9_\-=]+", i) for i in ids)  # Azure key rules


def test_document_without_acl_is_refused(tmp_path):
    (tmp_path / "a.md").write_text("# H\ntext")
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"documents": [{"doc_id": "a", "title": "A", "path": "a.md",
                                            "source_url": "u", "last_modified": "2025-01-01",
                                            "allowed_principals": []}]}))
    with pytest.raises(ValueError, match="no allowed_principals"):
        load_documents(m)


def test_all_sample_documents_load():
    docs = load_documents(DATA / "manifest.json")
    assert {d.file_type for d in docs} == {"docx", "pdf", "xlsx", "md"}
    assert all(d.sections for d in docs)
