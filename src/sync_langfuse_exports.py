#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
REPORT_SCRIPT = PROJECT_ROOT / "langfuse_export" / "build_nestful_run_report.py"

STEPS = [
    THIS_DIR / "export_langfuse_traces.py",
    THIS_DIR / "export_langfuse_observations.py",
    #THIS_DIR / "export_langfuse_scores.py",
    REPORT_SCRIPT,
]


def run_step(script_path: Path) -> None:
    print(f"\n=== Running {script_path.name} ===")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def main() -> None:
    for step in STEPS:
        run_step(step)

    print("\nLangfuse export sync complete.")
    print(f"Outputs written to: {PROJECT_ROOT / 'langfuse_export'}")


if __name__ == "__main__":
    main()
