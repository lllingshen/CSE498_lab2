#!/usr/bin/env python3
"""Compare ReVA metrics from Qwen-style and VILA-style evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_summary_line(lines: list[str], label: str) -> tuple[int, int, float] | None:
    pattern = re.compile(rf"^{re.escape(label)}:\s*(\d+)\s*/\s*(\d+)\s*=\s*([0-9.]+)%$")
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            numerator, denominator, percentage = match.groups()
            return int(numerator), int(denominator), float(percentage) / 100.0
    return None


def read_qwen_csv(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    total = parse_summary_line(lines, "Total")
    completed = parse_summary_line(lines, "Completed")
    num_questions = total[1] if total else None
    num_answered = completed[0] if completed else num_questions
    return {
        "accuracy": total[2] if total else None,
        "num_questions": num_questions,
        "num_answered": num_answered,
        "source": str(path),
    }


def read_vila_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "accuracy": data.get("accuracy"),
        "num_questions": data.get("num_questions"),
        "num_answered": data.get("num_answered"),
        "source": str(path),
    }


def format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-base",
        type=Path,
        default=Path("outputs/qwen_base/qwen_base/result.csv"),
        help="Base Qwen ReVA result.csv path.",
    )
    parser.add_argument(
        "--qwen-finetuned",
        type=Path,
        default=Path("outputs/qwen_sft/qwen_sft/result.csv"),
        help="Fine-tuned Qwen ReVA result.csv path.",
    )
    parser.add_argument(
        "--qwen",
        type=Path,
        default=None,
        help="Backward-compatible alias for --qwen-base.",
    )
    parser.add_argument(
        "--vila",
        type=Path,
        default=Path("outputs/vila_reva_v2/metrics.json"),
        help="VILA metrics.json path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/model_comparison.csv"),
        help="Output comparison CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qwen_base_path = args.qwen if args.qwen is not None else args.qwen_base
    rows = []

    if qwen_base_path.exists():
        metrics = read_qwen_csv(qwen_base_path)
        rows.append({"model": "Qwen base", **metrics})
    else:
        rows.append({"model": "Qwen base", "accuracy": None, "source": f"missing: {qwen_base_path}"})

    if args.qwen_finetuned.exists():
        metrics = read_qwen_csv(args.qwen_finetuned)
        rows.append({"model": "Qwen fine-tuned", **metrics})
    else:
        rows.append({"model": "Qwen fine-tuned", "accuracy": None, "source": f"missing: {args.qwen_finetuned}"})

    if args.vila.exists():
        metrics = read_vila_metrics(args.vila)
        rows.append(
            {
                "model": "VILA base",
                "accuracy": metrics["accuracy"],
                "source": metrics["source"],
                "num_questions": metrics.get("num_questions"),
                "num_answered": metrics.get("num_answered"),
            }
        )
    else:
        rows.append({"model": "VILA base", "accuracy": None, "source": f"missing: {args.vila}"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "num_questions", "num_answered", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "accuracy": format_pct(row.get("accuracy")),
                    "num_questions": row.get("num_questions", ""),
                    "num_answered": row.get("num_answered", ""),
                    "source": row["source"],
                }
            )

    for row in rows:
        print(f"{row['model']}: {format_pct(row.get('accuracy'))} ({row['source']})")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
