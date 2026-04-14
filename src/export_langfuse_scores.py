#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langfuse import Langfuse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "langfuse_export"
PAGE_SIZE = 100

# Export exactly one mode at a time by toggling these constants.
#PTC_FLAG = True
#OUTPUT_FILE = OUTPUT_DIR / "nestful_scores_ptc.jsonl"
PTC_FLAG = False
OUTPUT_FILE = OUTPUT_DIR / "nestful_scores_non_ptc.jsonl"

TRACE_EXPORT = {
    "label": "ptc" if PTC_FLAG else "regular",
    "tags": ["nestful", "ptc-fc"] if PTC_FLAG else ["nestful", "regular-fc"],
}


def safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def get_attr(obj: Any, attr_name: str, default: Any = None) -> Any:
    return getattr(obj, attr_name, default)


def normalize_score_row(score: Any) -> dict:
    trace_id = (
        get_attr(score, "trace_id")
        or get_attr(score, "traceId")
        or (
            get_attr(score, "trace").id
            if get_attr(score, "trace", None) is not None and hasattr(get_attr(score, "trace"), "id")
            else None
        )
    )

    observation_id = get_attr(score, "observation_id") or get_attr(score, "observationId")

    data_type = get_attr(score, "data_type") or get_attr(score, "dataType")
    value = get_attr(score, "value")
    if data_type == "CATEGORICAL" and value is not None:
        # Langfuse stores the human-readable category separately and `value`
        # can be a numeric category mapping (often 0.0 without a config).
        value = get_attr(score, "string_value") or get_attr(score, "stringValue") or value
    elif data_type == "BOOLEAN" and value is None:
        value = get_attr(score, "string_value") or get_attr(score, "stringValue")
    if value is None:
        value = get_attr(score, "numeric_value")
    if value is None:
        value = get_attr(score, "numericValue")
    if data_type == "CATEGORICAL" and value is None:
        value = get_attr(score, "string_value") or get_attr(score, "stringValue")

    row = {
        "id": get_attr(score, "id"),
        "traceId": trace_id,
        "observationId": observation_id,
        "name": get_attr(score, "name"),
        "value": value,
        "dataType": data_type,
        "comment": get_attr(score, "comment"),
        "createdAt": safe_str(get_attr(score, "created_at") or get_attr(score, "createdAt")),
        "updatedAt": safe_str(get_attr(score, "updated_at") or get_attr(score, "updatedAt")),
        "authorUserId": get_attr(score, "author_user_id") or get_attr(score, "authorUserId"),
        "configId": get_attr(score, "config_id") or get_attr(score, "configId"),
        "environment": get_attr(score, "environment"),
    }

    metadata = get_attr(score, "metadata", None)
    if metadata is not None:
        row["metadata"] = metadata

    return row


def fetch_trace_ids(langfuse: Langfuse, tags: list[str], label: str) -> set[str]:
    trace_ids: set[str] = set()
    page = 1

    while True:
        response = langfuse.api.trace.list(page=page, limit=PAGE_SIZE, tags=tags)
        traces = get_attr(response, "data")
        if traces is None:
            raise RuntimeError(
                f"Unexpected Langfuse trace response shape for {label} on page {page}. "
                "Expected an object with a .data field."
            )
        if not traces:
            break

        print(f"Fetched {label} traces page {page}: {len(traces)}")
        for trace in traces:
            trace_id = get_attr(trace, "id")
            if trace_id:
                trace_ids.add(trace_id)

        if len(traces) < PAGE_SIZE:
            break
        page += 1

    return trace_ids


def fetch_all_scores(langfuse: Langfuse) -> list[dict]:
    rows: list[dict] = []
    seen_score_ids: set[str] = set()
    page = 1

    while True:
        response = langfuse.api.scores.get_many(page=page, limit=PAGE_SIZE)
        #response = langfuse.api.scores.list(page=page, limit=PAGE_SIZE)
        scores = get_attr(response, "data")
        if scores is None:
            raise RuntimeError(
                f"Unexpected Langfuse score response shape on page {page}. "
                "Expected an object with a .data field."
            )
        if not scores:
            break

        print(f"Fetched scores page {page}: {len(scores)}")
        for score in scores:
            row = normalize_score_row(score)
            score_id = row.get("id")
            if score_id and score_id in seen_score_ids:
                continue
            if score_id:
                seen_score_ids.add(score_id)
            rows.append(row)

        if len(scores) < PAGE_SIZE:
            break
        page += 1

    return rows


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

    trace_ids = fetch_trace_ids(langfuse, TRACE_EXPORT["tags"], TRACE_EXPORT["label"])
    print(f"Found {len(trace_ids)} {TRACE_EXPORT['label']} trace ids")

    all_scores = fetch_all_scores(langfuse)
    filtered_scores = [row for row in all_scores if row.get("traceId") in trace_ids]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in filtered_scores:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(filtered_scores)} filtered scores to {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
