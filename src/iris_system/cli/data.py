from __future__ import annotations

import argparse
import json

from iris_system.data.manifest import compile_dataset, save_manifest
from iris_system.data.schemas import load_schema
from iris_system.data.validation import summarize_records, validate_records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compile and validate iris datasets")
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--domain", action="append")
    parser.add_argument("--schema", action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-hash", action="store_true")
    args = parser.parse_args(argv)
    records = []
    for index, root in enumerate(args.root):
        domain = args.domain[index] if args.domain and index < len(args.domain) else None
        schema = load_schema(args.schema[index]) if args.schema and index < len(args.schema) else None
        records.extend(compile_dataset(root, domain=domain, schema=schema, compute_hashes=not args.no_hash))
    issues = validate_records(records)
    print(json.dumps(summarize_records(records), ensure_ascii=False, indent=2))
    for issue in issues:
        print(f"[{issue.level}] {issue.sample_id}: {issue.message}")
    if any(issue.level == "error" for issue in issues):
        raise SystemExit("Dataset validation failed")
    save_manifest(records, args.output)
    print(f"manifest -> {args.output}")


if __name__ == "__main__":
    main()

