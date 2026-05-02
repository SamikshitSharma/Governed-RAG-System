from __future__ import annotations

from argparse import ArgumentParser
import sys

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs) -> bool:
        return False

from ingest import ingest_documents
from query import run_query
from config import ensure_directories


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Flat governed RAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest files from data/raw_documents")
    ingest_parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional override for the raw document directory.",
    )
    ingest_parser.add_argument(
        "--profile",
        default="generic",
        choices=["generic", "nda"],
        help="Ingestion profile to use.",
    )
    ingest_parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Clear the existing vector collection before ingesting.",
    )

    query_parser = subparsers.add_parser("query", help="Run a governed query")
    query_parser.add_argument("--question", required=True, help="Question to ask.")
    query_parser.add_argument("--user-id", required=True, help="Caller identifier.")
    query_parser.add_argument("--user-role", required=True, help="Caller role.")
    query_parser.add_argument("--top-k", type=int, default=None, help="Optional retrieval depth.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ensure_directories()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "ingest":
            ingest_kwargs = {
                "clear_existing": args.replace_existing,
                "profile": args.profile,
                "allowed_suffixes": {".pdf"} if args.profile == "nda" else None,
                "document_type": "NDA" if args.profile == "nda" else None,
                "sensitivity_level": "high" if args.profile == "nda" else None,
            }
            if args.source_dir:
                ingest_kwargs["source_dir"] = args.source_dir

            summary = ingest_documents(**ingest_kwargs)
            print("INGEST COMPLETE")
            print(f"Processed files: {summary['processed_files']}")
            print(f"Failed files: {summary['failed_files']}")
            print(f"Chunks stored: {summary['chunk_count']}")
            print(f"Backend: {summary['backend']}")
            print(f"Collection: {summary['collection_name']}")
            print(f"Batch ID: {summary['batch_id']}")
            return 0

        if args.command == "query":
            response = run_query(
                query=args.question,
                user_id=args.user_id,
                user_role=args.user_role,
                top_k=args.top_k,
            )
            print(f"Decision: {response.final_decision.value}")
            print(response.response_text)
            if response.retrieval_result.documents:
                print("\nSources:")
                for document in response.retrieval_result.documents:
                    print(f"- {document.metadata.source_file} ({document.metadata.authority})")
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
