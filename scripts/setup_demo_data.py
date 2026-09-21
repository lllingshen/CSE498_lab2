#!/usr/bin/env python3
"""Create a tiny runnable ReVA-style demo split from the bundled video."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_VIDEO_NAME = "v_7bUu05RIksU.mp4"
VIDEO_NAME = "demo_short.mp4"


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_reva_split(qa_items: list[dict]) -> dict:
    return {
        "videos": {
            "demo_video_001": {
                "file_path": f"videos/{VIDEO_NAME}",
                "dataset_name": "demo",
                "mcq": {
                    "video_qa": {
                        "visual_text_understanding": qa_items
                    }
                },
            }
        }
    }


def build_demo_train_split() -> dict:
    return build_reva_split(
        [
            {
                "qa_id": "DEMO-TRAIN-001",
                "question": "What type of business is advertised in the video title?",
                "options": {
                    "A": "A grocery store",
                    "B": "A hand car wash",
                    "C": "A railway station",
                    "D": "A hotel",
                },
                "correct_answer": "B",
                "reasoning": "The title card contains the words 'Premier Hand Car Wash'.",
            },
            {
                "qa_id": "DEMO-TRAIN-002",
                "question": "Which place name appears in the largest line of text?",
                "options": {
                    "A": "Oxford",
                    "B": "London",
                    "C": "Birmingham",
                    "D": "Bromsgrove",
                },
                "correct_answer": "D",
                "reasoning": "The largest title reads 'Bromsgrove Car Wash'.",
            },
        ]
    )


def build_demo_test_split() -> dict:
    return build_reva_split(
        [
            {
                "qa_id": "DEMO-TEST-001",
                "question": "Which road name is displayed on the title card?",
                "options": {
                    "A": "Oxford Road",
                    "B": "London Road",
                    "C": "Worcester Road",
                    "D": "Station Road",
                },
                "correct_answer": "C",
                "reasoning": "The address text includes 'Worcester Road'.",
            },
            {
                "qa_id": "DEMO-TEST-002",
                "question": "Which full service description appears below the main title?",
                "options": {
                    "A": "Premier Hand Car Wash",
                    "B": "Express Laundry Service",
                    "C": "Automatic Fuel Station",
                    "D": "Budget Bicycle Repair",
                },
                "correct_answer": "A",
                "reasoning": "The smaller text below the title reads 'Premier Hand Car Wash'.",
            },
        ]
    )


def copy_if_needed(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, target)


def make_short_video(source: Path, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-t",
        "2",
        "-vf",
        "scale='min(320,iw)':-2",
        "-an",
        str(target),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    source_video = PROJECT_ROOT / "data" / "qwen_train" / "videos" / SOURCE_VIDEO_NAME
    if not source_video.exists():
        source_video = PROJECT_ROOT / "qwen_finetune" / "demo" / "videos" / SOURCE_VIDEO_NAME
    if not source_video.exists():
        raise FileNotFoundError(f"Cannot find bundled demo video: {SOURCE_VIDEO_NAME}")

    train_video_dir = PROJECT_ROOT / "data" / "qwen_train" / "videos"
    test_video_dir = PROJECT_ROOT / "data" / "reva_test" / "videos"
    train_video_dir.mkdir(parents=True, exist_ok=True)
    test_video_dir.mkdir(parents=True, exist_ok=True)
    short_video = PROJECT_ROOT / "data" / "demo_reva" / VIDEO_NAME
    short_video.parent.mkdir(parents=True, exist_ok=True)
    make_short_video(source_video, short_video)
    copy_if_needed(short_video, train_video_dir / VIDEO_NAME)
    copy_if_needed(short_video, test_video_dir / VIDEO_NAME)

    train_split = build_demo_train_split()
    test_split = build_demo_test_split()
    write_json(PROJECT_ROOT / "data" / "demo_reva" / "train_set.json", train_split)
    write_json(PROJECT_ROOT / "data" / "demo_reva" / "test_set.json", test_split)
    write_json(PROJECT_ROOT / "data" / "reva_test" / "test_set.json", test_split)

    print("Demo data ready:")
    print(f"  train annotations: {PROJECT_ROOT / 'data/demo_reva/train_set.json'}")
    print(f"  test annotations:  {PROJECT_ROOT / 'data/reva_test/test_set.json'}")
    print(f"  train video:       {train_video_dir / VIDEO_NAME}")
    print(f"  test video:        {test_video_dir / VIDEO_NAME}")


if __name__ == "__main__":
    main()
