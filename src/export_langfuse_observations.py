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
OBSERVATION_EXPORTS = [
    {
        "label": "regular",
        "output_file": OUTPUT_DIR / "nestful_observations.jsonl",
        "tags": ["nestful", "regular-fc"],
    },
    {
        "label": "ptc",
        "output_file": OUTPUT_DIR / "nestful_observations_ptc.jsonl",
        "tags": ["nestful", "ptc-fc"],
    },
]
TRACE_PAGE_SIZE = 100
OBS_PAGE_SIZE = 100


def safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def get_attr(obj: Any, attr_name: str, default: Any = None) -> Any:
    return getattr(obj, attr_name, default)


def normalize_observation_row(obs: Any) -> dict:
    usage = get_attr(obs, "usage")
    usage_details = get_attr(obs, "usage_details") or get_attr(obs, "usageDetails") or {}
    input_tokens = get_attr(usage, "input")
    output_tokens = get_attr(usage, "output")
    total_tokens = (
        get_attr(usage, "total")
        if usage is not None
        else get_attr(obs, "totalTokens")
    )
    other_tokens = usage_details.get("thinking_tokens")
    if other_tokens is None and None not in (total_tokens, input_tokens, output_tokens):
        other_tokens = total_tokens - input_tokens - output_tokens

    return {
        "id": get_attr(obs, "id"),
        "traceId": get_attr(obs, "trace_id") or get_attr(obs, "traceId"),
        "parentObservationId": get_attr(obs, "parent_observation_id") or get_attr(obs, "parentObservationId"),
        "name": get_attr(obs, "name"),
        "type": get_attr(obs, "type"),
        "model": get_attr(obs, "model"),
        "version": get_attr(obs, "version"),
        "environment": get_attr(obs, "environment"),
        "input": get_attr(obs, "input"),
        "output": get_attr(obs, "output"),
        "metadata": get_attr(obs, "metadata"),
        "level": safe_str(get_attr(obs, "level")),
        "statusMessage": get_attr(obs, "status_message") or get_attr(obs, "statusMessage"),
        "latency": get_attr(obs, "latency"),
        "timeToFirstToken": get_attr(obs, "time_to_first_token") or get_attr(obs, "timeToFirstToken"),
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
        "usageDetails": usage_details,
        "costDetails": get_attr(obs, "cost_details") or get_attr(obs, "costDetails"),
        "inputTokens": input_tokens if input_tokens is not None else get_attr(obs, "promptTokens"),
        "outputTokens": output_tokens if output_tokens is not None else get_attr(obs, "completionTokens"),
        "totalTokens": total_tokens,
        "otherTokens": other_tokens,
        "promptTokens": get_attr(obs, "promptTokens"),
        "completionTokens": get_attr(obs, "completionTokens"),
        "calculatedInputCost": get_attr(obs, "calculated_input_cost") or get_attr(obs, "calculatedInputCost"),
        "calculatedOutputCost": get_attr(obs, "calculated_output_cost") or get_attr(obs, "calculatedOutputCost"),
        "calculatedTotalCost": get_attr(obs, "calculated_total_cost") or get_attr(obs, "calculatedTotalCost"),
        "startTime": safe_str(get_attr(obs, "start_time") or get_attr(obs, "startTime")),
        "endTime": safe_str(get_attr(obs, "end_time") or get_attr(obs, "endTime")),
        "completionStartTime": safe_str(get_attr(obs, "completion_start_time") or get_attr(obs, "completionStartTime")),
        "createdAt": safe_str(get_attr(obs, "createdAt") or get_attr(obs, "created_at")),
        "updatedAt": safe_str(get_attr(obs, "updatedAt") or get_attr(obs, "updated_at")),
    }


def fetch_trace_ids(langfuse: Langfuse, tags: list[str], label: str) -> list[str]:
    trace_ids: list[str] = []
    page = 1

    while True:
        try:
            response = langfuse.api.trace.list(page=page, limit=TRACE_PAGE_SIZE, tags=tags)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch Langfuse traces for {label} on page {page}: {e}"
            ) from e

        traces = get_attr(response, "data")
        if traces is None:
            raise RuntimeError(
                f"Unexpected Langfuse trace response shape for {label} on page {page}. "
                "Expected an object with a .data field."
            )

        if not traces:
            break

        print(f"Fetched {label} trace page {page}: {len(traces)}")
        trace_ids.extend([trace.id for trace in traces if get_attr(trace, "id")])

        if len(traces) < TRACE_PAGE_SIZE:
            break

        page += 1

    return trace_ids


def export_observations_for_traces(langfuse: Langfuse, trace_ids: list[str], output_file: Path, label: str) -> None:
    total_written = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, trace_id in enumerate(trace_ids, start=1):
            page = 1
            while True:
                try:
                    response = langfuse.api.legacy.observations_v1.get_many(
                        page=page,
                        limit=OBS_PAGE_SIZE,
                        trace_id=trace_id,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to fetch observations for {label} trace {trace_id} on page {page}: {e}"
                    ) from e

                observations = get_attr(response, "data")
                if observations is None:
                    raise RuntimeError(
                        f"Unexpected Langfuse observation response shape for {label} trace {trace_id} on page {page}. "
                        "Expected an object with a .data field."
                    )

                if not observations:
                    break

                for obs in observations:
                    row = normalize_observation_row(obs)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_written += 1

                if len(observations) < OBS_PAGE_SIZE:
                    break

                page += 1

            if idx % 25 == 0 or idx == len(trace_ids):
                print(f"Exported {label} observations for {idx}/{len(trace_ids)} traces")

    print(f"Saved {total_written} {label} observations to {output_file}")


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

    for export_cfg in OBSERVATION_EXPORTS:
        trace_ids = fetch_trace_ids(langfuse, export_cfg["tags"], export_cfg["label"])
        print(f"Found {len(trace_ids)} {export_cfg['label']} traces")
        export_observations_for_traces(
            langfuse=langfuse,
            trace_ids=trace_ids,
            output_file=export_cfg["output_file"],
            label=export_cfg["label"],
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
