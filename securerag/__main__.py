"""Command line interface.

    python -m securerag ingest                 # build the index (Azure mode)
    python -m securerag ask --user ayesha "How many days of annual leave do I get?"
    python -m securerag eval                   # retrieval + security evaluation
"""

from __future__ import annotations

import argparse
import json
import sys

from securerag.config import Settings
from securerag.directory import JsonDirectory


def _print_answer(a: dict) -> None:
    print(f"\n[AI-generated answer | status: {a['status']}]\n")
    print(a["answer"])
    for n in a["notes"]:
        print(f"\n  ! {n}")
    if a["citations"]:
        print("\nSources:")
        for c in a["citations"]:
            page = f", page {c['page']}" if c.get("page") else ""
            print(f"  [{c['ref']}] {c['title']} > {c['section']}{page} "
                  f"(updated {c['last_modified']})\n      {c['source_url']}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="securerag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Parse, chunk, embed and index the document library")
    p_ask = sub.add_parser("ask", help="Ask a question as a given user")
    p_ask.add_argument("--user", required=True)
    p_ask.add_argument("--json", action="store_true")
    p_ask.add_argument("question")
    p_eval = sub.add_parser("eval", help="Run the retrieval/security evaluation")
    p_eval.add_argument("--k", type=int, default=5)
    p_eval.add_argument("--out", default="eval/results.md")
    args = parser.parse_args(argv)

    s = Settings.from_env()

    if args.cmd == "ingest":
        from securerag.factory import build_backend, build_embedder, ingest
        if s.backend == "azure":
            s.require_azure()
        n = ingest(s, build_backend(s), build_embedder(s))
        print(f"Indexed {n} chunks into the {s.backend} backend.")
        if s.backend == "local":
            print("(Local mode keeps the index in memory; `ask` and `eval` rebuild it automatically.)")
        return 0

    if args.cmd == "ask":
        from securerag.factory import build_pipeline
        user = JsonDirectory(s.directory).get_user(args.user)
        answer = build_pipeline(s).ask(args.question, user).to_dict()
        print(json.dumps(answer, indent=2)) if args.json else _print_answer(answer)
        return 0

    if args.cmd == "eval":
        from eval.run_eval import run
        return run(s, k=args.k, out=args.out)
    return 1


if __name__ == "__main__":
    sys.exit(main())
