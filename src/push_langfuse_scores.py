#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse


PAGE_SIZE = 100


def read_jsonl(path):
    rows = []
    input_path = Path(path)
    if not input_path.exists():
        return rows
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_trace_attr(trace, key, default=None):
    metadata = getattr(trace, "metadata", None) or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    attributes = metadata.get("attributes", {}) if isinstance(metadata, dict) else {}
    return attributes.get(key, default)


def get_trace_tags(trace):
    trace_tags = getattr(trace, "tags", None) or []
    if trace_tags:
        return trace_tags

    metadata = getattr(trace, "metadata", None) or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    attributes = metadata.get("attributes", {}) if isinstance(metadata, dict) else {}
    raw_tags = attributes.get("langfuse.trace.tags", [])
    if isinstance(raw_tags, str):
        try:
            raw_tags = json.loads(raw_tags)
        except Exception:
            raw_tags = []
    return raw_tags if isinstance(raw_tags, list) else []


def get_trace_timestamp(trace):
    for attr_name in ("timestamp", "created_at", "createdAt", "start_time", "startTime"):
        value = getattr(trace, attr_name, None)
        if value:
            return str(value)
    return ""


def make_trace_tags(sample_id, ptc_enabled, model_name, model_provider):
    ptc_flag = "ptc-fc" if ptc_enabled else "regular-fc"
    return [
        "nestful",
        sample_id,
        ptc_flag,
        model_name,
        model_provider,
    ]


def fetch_candidate_traces(langfuse, server_tag_filter, ptc_enabled):
    if server_tag_filter == "mode":
        tags = ["nestful", "ptc-fc" if ptc_enabled else "regular-fc"]
    elif server_tag_filter == "benchmark":
        tags = ["nestful"]
    else:
        tags = None

    traces = []
    page = 1
    while True:
        kwargs = {"page": page, "limit": PAGE_SIZE}
        if tags:
            kwargs["tags"] = tags
        response = langfuse.api.trace.list(**kwargs)
        batch = getattr(response, "data", None)
        if batch is None:
            raise RuntimeError("Unexpected Langfuse trace response shape; expected .data")
        if not batch:
            break
        traces.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        page += 1
    return traces


def build_trace_index(traces):
    by_sample_id = {}
    for trace in traces:
        sample_id = get_trace_attr(trace, "benchmark.sample_id", getattr(trace, "name", None))
        if not sample_id:
            continue
        by_sample_id.setdefault(sample_id, []).append(trace)
    return by_sample_id


def select_trace(record, candidates):
    sample_id = record["sample_id"]
    expected_ptc = "true" if record["ptc_enabled"] else "false"
    model_name = record["toolman_model_name"]
    model_provider = record["toolman_model_provider"]
    expected_tags = set(make_trace_tags(sample_id, record["ptc_enabled"], model_name, model_provider))
    accepted_models = {
        model_name,
        f"{model_provider}/{model_name}",
    }

    sample_matches = [trace for trace in candidates if get_trace_attr(trace, "benchmark.sample_id", None) == sample_id]
    ptc_matches = [trace for trace in sample_matches if str(get_trace_attr(trace, "ptc.enabled", "")).lower() == expected_ptc]
    model_matches = [trace for trace in ptc_matches if get_trace_attr(trace, "gen_ai.request.model", "") in accepted_models]

    if len(model_matches) == 1:
        return model_matches[0], "sample_id+ptc+model"

    if len(model_matches) > 1:
        tag_matches = [trace for trace in model_matches if expected_tags.issubset(set(get_trace_tags(trace)))]
        if len(tag_matches) == 1:
            return tag_matches[0], "sample_id+ptc+model+tags"
        if tag_matches:
            tag_matches.sort(key=get_trace_timestamp, reverse=True)
            return tag_matches[0], "sample_id+ptc+model+tags+latest"
        model_matches.sort(key=get_trace_timestamp, reverse=True)
        return model_matches[0], "sample_id+ptc+model+latest"

    return None, "no_match"


def add_langfuse_scores(langfuse, trace_id, record):
    sample_id = record["sample_id"]
    unexpected_tools = record.get("unexpected_tools") or []
    unexpected_arguments = record.get("unexpected_arguments") or []

    langfuse.create_score(
        trace_id=trace_id,
        name="nestful_exact_match",
        value=1.0 if record["cmp_type"] == "exact_match" else 0.0,
        data_type="NUMERIC",
        comment=f"sample_id={sample_id}",
    )
    langfuse.create_score(
        trace_id=trace_id,
        name="nestful_partial_match_accuracy",
        value=float(record["accuracy_combined"]),
        data_type="NUMERIC",
        comment=f"sample_id={sample_id}",
    )
    langfuse.create_score(
        trace_id=trace_id,
        name="nestful_error_type",
        value=record["cmp_type"],
        data_type="CATEGORICAL",
        comment=f"sample_id={sample_id}",
    )
    langfuse.create_score(
        trace_id=trace_id,
        name="nestful_win",
        value=1.0 if record["win_score"] else 0.0,
        data_type="NUMERIC",
        comment=f"sample_id={sample_id}",
    )
    langfuse.create_score(
        trace_id=trace_id,
        name="pred_contains_unexpected_tool",
        value=1.0 if unexpected_tools else 0.0,
        data_type="NUMERIC",
        comment=f"sample_id={sample_id}; unexpected_tools={unexpected_tools}",
    )
    langfuse.create_score(
        trace_id=trace_id,
        name="pred_contains_unexpected_argument",
        value=1.0 if unexpected_arguments else 0.0,
        data_type="NUMERIC",
        comment=f"sample_id={sample_id}; unexpected_arguments={unexpected_arguments}",
    )


def load_successful_sample_ids(path):
    if not path:
        return set()
    rows = read_jsonl(path)
    return {
        row.get("sample_id")
        for row in rows
        if row.get("status") == "pushed" and row.get("sample_id")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring_records_path", type=str, required=True)
    parser.add_argument("--push_results_output_path", type=str, required=True)
    parser.add_argument("--ptc_enabled", action="store_true")
    parser.add_argument("--skip_successes_from", type=str, default=None)
    parser.add_argument("--server_tag_filter", choices=["none", "benchmark", "mode"], default="none")
    args = parser.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL")
    if not public_key or not secret_key or not host:
        raise ValueError("Missing Langfuse environment variables in .env")

    scoring_records = read_jsonl(args.scoring_records_path)
    previous_status_path = args.skip_successes_from or (
        args.push_results_output_path if Path(args.push_results_output_path).exists() else None
    )
    skip_sample_ids = load_successful_sample_ids(previous_status_path)

    langfuse = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )

    traces = fetch_candidate_traces(
        langfuse,
        server_tag_filter=args.server_tag_filter,
        ptc_enabled=args.ptc_enabled,
    )
    trace_index = build_trace_index(traces)

    results = []
    pushed_count = 0
    skipped_count = 0
    not_found_count = 0
    score_error_count = 0

    for record in scoring_records:
        sample_id = record.get("sample_id")
        if not sample_id:
            results.append({"sample_id": None, "status": "missing_sample_id"})
            continue

        if sample_id in skip_sample_ids:
            skipped_count += 1
            results.append({
                "sample_id": sample_id,
                "status": "skipped_existing_success",
            })
            continue

        candidates = trace_index.get(sample_id, [])
        trace, lookup_strategy = select_trace(record, candidates)
        if trace is None:
            not_found_count += 1
            results.append({
                "sample_id": sample_id,
                "status": "trace_not_found",
                "lookup_strategy": lookup_strategy,
                "candidate_count": len(candidates),
            })
            continue

        trace_id = getattr(trace, "id", None)
        try:
            add_langfuse_scores(langfuse, trace_id, record)
            pushed_count += 1
            results.append({
                "sample_id": sample_id,
                "status": "pushed",
                "trace_id": trace_id,
                "lookup_strategy": lookup_strategy,
                "candidate_count": len(candidates),
            })
        except Exception as e:
            score_error_count += 1
            results.append({
                "sample_id": sample_id,
                "status": "score_error",
                "trace_id": trace_id,
                "lookup_strategy": lookup_strategy,
                "candidate_count": len(candidates),
                "error": str(e),
            })

    write_jsonl(args.push_results_output_path, results)
    langfuse.flush()
    langfuse.shutdown()

    print(f"Total scoring records: {len(scoring_records)}")
    print(f"Traces indexed: {len(traces)}")
    print(f"Server tag filter: {args.server_tag_filter}")
    print(f"Pushed: {pushed_count}")
    print(f"Skipped existing successes: {skipped_count}")
    print(f"Trace not found: {not_found_count}")
    print(f"Score errors: {score_error_count}")
    print(f"Saved push results to: {args.push_results_output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
