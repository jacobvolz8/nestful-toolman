#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langfuse import Langfuse


OUTPUT_DIR = Path("langfuse_export")
TRACE_EXPORTS = [
    {
        "label": "regular",
        "output_file": OUTPUT_DIR / "nestful_traces.jsonl",
        "tags": ["nestful", "regular-fc"],
    },
    {
        "label": "ptc",
        "output_file": OUTPUT_DIR / "nestful_traces_ptc.jsonl",
        "tags": ["nestful", "ptc-fc"],
    },
]
PAGE_SIZE = 100


def safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def get_attr(obj: Any, attr_name: str, default: Any = None) -> Any:
    return getattr(obj, attr_name, default)


def normalize_trace_row(trace: Any) -> dict:
    row = {
        "id": get_attr(trace, "id"),
        "name": get_attr(trace, "name"),
        "session_id": get_attr(trace, "session_id") or get_attr(trace, "sessionId"),
        "user_id": get_attr(trace, "user_id") or get_attr(trace, "userId"),
        "environment": get_attr(trace, "environment"),
        "latency": get_attr(trace, "latency"),
        "total_cost": get_attr(trace, "total_cost") or get_attr(trace, "totalCost"),
        "input": get_attr(trace, "input"),
        "output": get_attr(trace, "output"),
        "metadata": get_attr(trace, "metadata"),
        "tags": get_attr(trace, "tags"),
        "scores": get_attr(trace, "scores"),
        "observations": get_attr(trace, "observations"),
        "timestamp": safe_str(get_attr(trace, "timestamp")),
        "createdAt": safe_str(get_attr(trace, "createdAt") or get_attr(trace, "created_at")),
        "updatedAt": safe_str(get_attr(trace, "updatedAt") or get_attr(trace, "updated_at")),
        "release": get_attr(trace, "release"),
        "version": get_attr(trace, "version"),
        "public": get_attr(trace, "public"),
        "externalId": get_attr(trace, "externalId") or get_attr(trace, "external_id"),
    }
    return row


def export_traces(langfuse: Langfuse, output_file: Path, tags: list[str], label: str) -> None:
    total_written = 0
    page = 1

    with open(output_file, "w", encoding="utf-8") as f:
        while True:
            try:
                response = langfuse.api.trace.list(
                    page=page,
                    limit=PAGE_SIZE,
                    tags=tags,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to fetch Langfuse traces for {label} on page {page}: {e}"
                ) from e

            traces = get_attr(response, "data")
            if traces is None:
                raise RuntimeError(
                    f"Unexpected Langfuse response shape for {label} on page {page}. "
                    "Expected an object with a .data field."
                )

            if not traces:
                break

            print(f"Fetched {label} traces page {page}: {len(traces)}")

            for trace in traces:
                row = normalize_trace_row(trace)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_written += 1

            if len(traces) < PAGE_SIZE:
                break

            page += 1

    print(f"Saved {total_written} {label} traces to {output_file}")


def main() -> None:
    load_dotenv()

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL")

    if not public_key or not secret_key or not host:
        raise ValueError(
            "Missing Langfuse environment variables. "
            "Expected LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL in .env"
        )

    langfuse = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for trace_export in TRACE_EXPORTS:
        export_traces(
            langfuse=langfuse,
            output_file=trace_export["output_file"],
            tags=trace_export["tags"],
            label=trace_export["label"],
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
