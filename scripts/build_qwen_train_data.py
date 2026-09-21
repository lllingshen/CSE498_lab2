#!/usr/bin/env python3
"""Convert ReVA annotations into Qwen-VL conversation training data.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_video_path(file_path: str, video_root: Path | None = None) -> str:
    path = Path(file_path)
    if video_root is None:
        return path.as_posix()
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(video_root).as_posix()
    except ValueError:
        return path.as_posix()


def format_options(options: dict[str, str]) -> str:
    """Return sorted multiple-choice options, one option per line."""
    return "\n".join(f"{label}. {options[label]}" for label in sorted(options))


def format_question(question: str, options: dict[str, str], prompt_style: str) -> str:
    """Build the human message used by Qwen SFT."""
    text = f"Question: {question}\n{format_options(options)}"
    if prompt_style == "plain":
        return f"<video>\n{text}\nAnswer with only the option letter from the given choices."
    if prompt_style == "reva_eval":
        return (
            "<video>\nPlease carefully watch the video and answer the multiple-choice question below.\n"
            "Think step-by-step within <think> </think> tags, then provide only the letter of the "
            f"correct option within <answer> </answer> tags.\n{text}"
        )
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def format_answer(qa: dict[str, Any], answer_style: str) -> str:
    """Format the assistant target from a ReVA QA item."""
    answer = str(qa["correct_answer"]).strip().upper()
    if answer_style == "letter":
        return f"Answer: {answer}"
    if answer_style == "tagged":
        return f"<answer>{answer}</answer>"
    if answer_style == "cot_tagged":
        return f"<think>{qa.get('reasoning', '')}</think><answer>{answer}</answer>"
    raise ValueError(f"Unknown answer style: {answer_style}")


def iter_reva_qas(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one flat QA record at a time from nested ReVA annotations."""
    for video_id, video_item in data.get("videos", {}).items():
        for category, question_types in video_item.get("mcq", {}).items():
            for question_type, qas in question_types.items():
                for qa in qas:
                    yield {
                        "video_id": video_id,
                        "file_path": video_item["file_path"],
                        "dataset_name": video_item.get("dataset_name"),
                        "category": category,
                        "question_type": question_type,
                        "qa": qa,
                    }


def convert_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """Convert one flat ReVA QA item into one Qwen conversation sample."""
    file_path = str(item["file_path"]).replace("\\", "/").removeprefix("./")
    file_path = file_path.removeprefix("#dataset/").removeprefix("ReVA_V2/")
    video = normalize_video_path(file_path, args.video_root)
    if args.require_video and not (args.video_root / video).is_file():
        return None
    qa = item["qa"]
    sample = {
        "video": video,
        "conversations": [
            {"from": "human", "value": format_question(qa["question"], qa["options"], args.prompt_style)},
            {"from": "gpt", "value": format_answer(qa, args.answer_style)},
        ],
    }
    if args.keep_metadata:
        sample["metadata"] = {
            "video_id": item["video_id"],
            "qa_id": qa.get("qa_id"),
            "global_index": qa.get("global_index"),
            "dataset_name": item.get("dataset_name"),
            "category": item["category"],
            "question_type": item["question_type"],
        }
    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/reva_train/train_set.json"))
    parser.add_argument("--output", type=Path, default=Path("data/qwen_train/train.json"))
    parser.add_argument("--video-root", type=Path, default=Path("data/qwen_train"))
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all.")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all.")
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--prompt-style", choices=["plain", "reva_eval"], default="reva_eval")
    parser.add_argument("--answer-style", choices=["letter", "tagged", "cot_tagged"], default="tagged")
    parser.add_argument("--keep-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input)

    samples = []
    seen_videos = set()
    skipped_missing_video = 0
    for item in iter_reva_qas(data):
        seen_videos.add(item["video_id"])
        if args.max_videos and len(seen_videos) > args.max_videos:
            break
        sample = convert_item(item, args)
        if sample is None:
            skipped_missing_video += 1
            continue
        samples.append(sample)
        if args.max_samples and len(samples) >= args.max_samples:
            break

    if args.require_video and not samples:
        raise SystemExit(
            "No training samples reference an existing video. "
            "Check QWEN_VIDEO_ROOT and the downloaded ReVA directory; output was not modified."
        )

    dump_json(samples, args.output)
    print(f"Saved {len(samples)} Qwen training samples to {args.output}")
    if skipped_missing_video:
        print(f"Skipped {skipped_missing_video} samples with missing videos")


if __name__ == "__main__":
    main()
