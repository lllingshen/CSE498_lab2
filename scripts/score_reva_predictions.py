#!/usr/bin/env python3
"""Score Qwen-style ReVA prediction JSONL files.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_answer(text: str) -> str:
    """Extract text inside <answer>...</answer>; fall back to full text."""
    text = str(text or "")
    answers = re.findall(r"<answer\b[^>]*>(.*?)</answer\s*>", text, re.I | re.S)
    return (answers[-1] if answers else text).strip()


def extract_letter(text: str) -> str:
    """Extract one answer letter A-H from model output."""
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", extract_answer(text), flags=re.I | re.S).strip()
    for prefix in (r"final\s+answer", r"answer|correct\s+(?:choice|option)|(?:choose|select|pick)"):
        matches = [match for match in re.finditer(
            rf"\b(?:{prefix})\s*(?:is\s*|:\s*|=\s*)?(?:option\s*)?[\s(*`\[\"']*([A-H])\b",
            text, re.I,
        ) if not re.search(r"\b(?:not|never|don['’]t)\s*[*_`]*\s*$", text[:match.start()], re.I)]
        if matches:
            match = matches[-1]
            if re.match(r"[\s)\]*_`\"']*(?:or\b|/|and\b)[\s([*_`\"']*[A-H]\b", text[match.end():], re.I):
                return ""
            if match.group(1) == "a" and re.match(r"\s+[a-z]", text[match.end():]):
                continue
            return match.group(1).upper()
    match = re.fullmatch(r"(?:option\s+)?[\s\W]*([A-H])[\s\W]*", text, re.I)
    if not match:
        match = re.match(r"^\s*(?:option\s+)?[(*`\[]*([A-H])[.)\]:-](?:\s|$)", text, re.I)
    if match and re.match(r"[\s)\]*_`\"']*(?:or\b|/|and\b)[\s([*_`\"']*[A-H]\b", text[match.end():], re.I):
        return ""
    return match.group(1).upper() if match else ""


def load_predictions(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prediction JSONL files from output_dir, ignoring result.json."""
    predictions = {}
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "result.json":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict) and "id" in item:
                    predictions[str(item["id"])] = item
    return predictions


def score_predictions(preds: dict[str, dict[str, Any]], gt_list: list[dict[str, Any]]) -> tuple[dict, str]:
    """Score every GT item and report completion separately from accuracy."""
    results, groups = {}, defaultdict(lambda: [0, 0])
    completed = correct = 0
    for gt in gt_list:
        qa_id = str(gt["id"])
        if qa_id in results:
            raise ValueError(f"Duplicate ground-truth ID: {qa_id}")
        pred = preds.get(qa_id, {})
        text = str(pred.get("pred") or "")
        letter = extract_letter(text)
        options = gt.get("options") or dict(re.findall(r"(?m)^\s*([A-H])[.):]\s*(.+)$", gt.get("question", "")))
        options = {str(key).upper(): str(value) for key, value in options.items()}
        if not letter and options:
            answer_text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", extract_answer(text), flags=re.I | re.S)
            answer_text = re.split(r"\b(?:final\s+answer|answer)\s*(?::|=|\bis\b)\s*", answer_text, flags=re.I)[-1]
            normalized = re.sub(r"\W+", " ", answer_text.casefold()).strip()
            normalized = re.sub(r"^(?:(?:it|this) is|(?:the )?(?:correct )?(?:answer|choice|option) is)\s+", "", normalized)
            normalized = re.sub(r"^(?:a|an|the)\s+", "", normalized)
            matches = []
            for key, value in options.items():
                value = re.sub(r"\W+", " ", value.casefold()).strip()
                value = re.sub(r"^(?:a|an|the)\s+", "", value)
                if value and normalized == value:
                    matches.append(key)
            letter = matches[0] if len(matches) == 1 else ""
        if options and letter not in options:
            letter = ""
        answer = extract_letter(gt.get("answer", ""))
        acc = int(bool(letter) and letter == answer)
        completed += bool(letter)
        correct += acc
        group = gt.get("question_type", gt.get("subcategory", "unknown"))
        groups[group][0] += acc
        groups[group][1] += 1
        results[qa_id] = {**pred, **gt, "pred": text, "pred_letter": letter, "acc": acc,
                          "prediction_present": qa_id in preds}
    total = len(gt_list)
    lines = [f"Completed: {completed}/{total} = {100 * completed / total if total else 0:.2f}%",
             f"Total: {correct}/{total} = {100 * correct / total if total else 0:.2f}%"]
    lines.extend(f"{name}: {hits}/{count} = {100 * hits / count:.2f}%"
                 for name, (hits, count) in sorted(groups.items()))
    return results, "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = load_predictions(args.output_dir)
    gt_list = json.loads(args.gt_file.read_text(encoding="utf-8"))
    results, csv_text = score_predictions(preds, gt_list)

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Results ===")
    print(csv_text)

    csv_path = args.output_dir / "result.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
