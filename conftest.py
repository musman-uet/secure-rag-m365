import json
import os
from pathlib import Path

import pytest

from securerag.audit import AuditLog
from securerag.directory import JsonDirectory
from securerag.embeddings import HashingEmbedder
from securerag.generation import ExtractiveGenerator
from securerag.ingest.loader import load_chunks
from securerag.pipeline import RagPipeline
from securerag.search.local_backend import LocalSearchBackend

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(scope="session", autouse=True)
def sample_data():
    if not (DATA / "manifest.json").exists():
        import runpy
        runpy.run_path(str(ROOT / "scripts" / "make_sample_data.py"), run_name="__main__")
    os.environ.setdefault("RAG_BACKEND", "local")


@pytest.fixture(scope="session")
def manifest():
    return json.loads((DATA / "manifest.json").read_text())


@pytest.fixture(scope="session")
def acl(manifest):
    return {d["doc_id"]: set(d["allowed_principals"]) for d in manifest["documents"]}


@pytest.fixture(scope="session")
def directory():
    return JsonDirectory(DATA / "directory.json")


@pytest.fixture(scope="session")
def indexed():
    embedder = HashingEmbedder()
    chunks = load_chunks(DATA / "manifest.json")
    for c, v in zip(chunks, embedder.embed([c.content for c in chunks])):
        c.vector = v
    backend = LocalSearchBackend()
    backend.index(chunks)
    return backend, embedder, chunks


@pytest.fixture
def pipeline(indexed, tmp_path):
    backend, embedder, _ = indexed
    return RagPipeline(backend, embedder, ExtractiveGenerator(), AuditLog(tmp_path / "audit.jsonl"))
