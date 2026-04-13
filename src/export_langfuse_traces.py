import os
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv(Path(__file__).with_name(".env"))

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

def to_plain_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return json.loads(json.dumps(obj, default=str))

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

def fetch_all_traces(tags=None, page_limit=100):
    all_traces = []
    page = 1

    while True:
        res = langfuse.api.trace.list(
            page=page,
            limit=page_limit,
            tags=tags,
        )
        traces = getattr(res, "data", []) or []

        if not traces:
            break

        all_traces.extend(traces)
        print(f"Fetched traces page {page}: {len(traces)}")

        if len(traces) < page_limit:
            break

        page += 1
        time.sleep(0.1)

    return all_traces

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="langfuse_export")
    parser.add_argument("--page_limit", type=int, default=100)
    parser.add_argument("--extra_tags", nargs="*", default=[])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tags = ["nestful"] + args.extra_tags
    traces = fetch_all_traces(tags=tags, page_limit=args.page_limit)
    trace_dicts = [to_plain_dict(t) for t in traces]

    traces_path = output_dir / "nestful_traces.jsonl"
    write_jsonl(traces_path, trace_dicts)

    print(f"Saved {len(trace_dicts)} traces to {traces_path}")

if __name__ == "__main__":
    main()