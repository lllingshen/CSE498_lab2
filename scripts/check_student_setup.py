#!/usr/bin/env python3
"""Check the files and commands needed for the student workflow."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def configured_path(name: str, default: Path) -> Path:
    path = Path(os.environ.get(name) or default).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def status(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def check_file(path: Path, note: str) -> bool:
    ok = path.is_file() and path.stat().st_size > 0
    print(f"[{status(ok):7}] {display_path(path)} - {note}")
    return ok


def check_dir(path: Path, note: str) -> bool:
    ok = path.is_dir()
    print(f"[{status(ok):7}] {display_path(path)} - {note}")
    return ok


def count_json_items(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and isinstance(data.get("videos"), dict):
        if any(
            not isinstance(item, dict) or not isinstance(item.get("file_path"), str)
            or not item["file_path"].strip() for item in data["videos"].values()
        ):
            return None
        return len(data["videos"])
    return None


def resolve_video_reference(root: Path, reference: str) -> Path | None:
    normalized = reference.replace("\\", "/").removeprefix("./")
    if not normalized.strip():
        return None
    path = Path(normalized)
    candidates = [path] if path.is_absolute() else [root / path]

    for prefix in ("#dataset/ReVA_V2/", "#dataset/ReVA/", "ReVA_V2/", "ReVA/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            candidates.append(root / normalized)
            break

    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
        if candidate.is_dir() and any(
            frame.is_file() and frame.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            and frame.stat().st_size > 0 for frame in candidate.iterdir()
        ):
            return candidate
    return None


def check_qwen_video_references(annotation_path: Path, video_root: Path) -> bool:
    try:
        samples = json.loads(annotation_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[MISSING] could not read {display_path(annotation_path)}: {exc}")
        return False
    if not isinstance(samples, list) or not samples:
        print(f"[MISSING] {display_path(annotation_path)} contains no training samples")
        return False

    if any(
        not isinstance(sample, dict) or not isinstance(sample.get("video"), str)
        or not sample["video"].strip() for sample in samples
    ):
        print(f"[MISSING] {display_path(annotation_path)} has a missing or invalid video reference")
        return False
    references = sorted({sample["video"] for sample in samples})
    missing = [reference for reference in references if resolve_video_reference(video_root, reference) is None]
    if missing:
        print(
            f"[MISSING] {len(missing)}/{len(references)} Qwen training video references are missing or empty "
            f"under {display_path(video_root)}"
        )
        for reference in missing[:5]:
            print(f"          {reference}")
        return False
    print(f"[OK     ] resolved all {len(references)} Qwen training video references")
    return True


def check_reva_video_references(annotation_path: Path, video_root: Path) -> bool:
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        videos = data["videos"]
        if not isinstance(videos, dict) or not videos:
            raise ValueError("contains no video records")
        if any(
            not isinstance(item, dict) or not isinstance(item.get("file_path"), str)
            or not item["file_path"].strip() for item in videos.values()
        ):
            raise ValueError("has a missing or invalid file_path")
    except Exception as exc:
        print(f"[MISSING] could not read {display_path(annotation_path)}: {exc}")
        return False

    references = sorted({item["file_path"] for item in videos.values()})
    missing = [reference for reference in references if resolve_video_reference(video_root, reference) is None]
    if missing:
        print(
            f"[MISSING] {len(missing)}/{len(references)} ReVA test video references are missing or empty "
            f"under {display_path(video_root)}"
        )
        for reference in missing[:5]:
            print(f"          {reference}")
        return False
    print(f"[OK     ] resolved all {len(references)} ReVA test video references")
    return True


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print("")

    reva_root = configured_path("REVA_ROOT", PROJECT_ROOT / "data/reva_test")
    reva_json = configured_path("REVA_JSON", reva_root / "test_set.json")
    train_json = configured_path("REVA_TRAIN_JSON", PROJECT_ROOT / "data/reva_train/train_set.json")
    qwen_json = configured_path("QWEN_TRAIN_JSON", PROJECT_ROOT / "data/qwen_train/train.json")
    qwen_root = configured_path("QWEN_VIDEO_ROOT", PROJECT_ROOT / "data/qwen_train")
    required = [
        (train_json, "source annotations for training-data conversion"),
        (qwen_json, "Qwen-format SFT data"),
        (reva_json, "ReVA evaluation questions"),
        (PROJECT_ROOT / "scripts/prepare_qwen_train_data.sh", "data conversion entry point"),
        (PROJECT_ROOT / "scripts/run_eval_qwen_base.sh", "base Qwen evaluation entry point"),
        (PROJECT_ROOT / "scripts/run_finetune_qwen.sh", "Qwen SFT/LoRA entry point"),
        (PROJECT_ROOT / "scripts/run_eval_qwen_finetuned.sh", "fine-tuned Qwen evaluation entry point"),
        (PROJECT_ROOT / "scripts/run_eval_reva_vila.sh", "VILA evaluation entry point"),
        (PROJECT_ROOT / "scripts/compare_models.sh", "metric comparison entry point"),
    ]

    all_ok = True
    for path, note in required:
        all_ok = check_file(path, note) and all_ok
    all_ok = check_dir(qwen_root, "Qwen training video root") and all_ok
    all_ok = check_dir(reva_root, "ReVA evaluation video root") and all_ok
    all_ok = check_qwen_video_references(qwen_json, qwen_root) and all_ok
    all_ok = check_reva_video_references(reva_json, reva_root) and all_ok

    print("")
    for path in [train_json, qwen_json, reva_json]:
        count = count_json_items(path)
        if count:
            print(f"[INFO   ] {display_path(path)} contains {count} top-level items")
        else:
            print(f"[MISSING] {display_path(path)} has no valid annotation items")
            all_ok = False

    print("")
    conda_env = os.environ.get("CONDA_ENV", "qwen2")
    conda_bin = os.environ.get("CONDA_BIN") or "conda"
    commands = ["python3"] + ([conda_bin] if conda_env else [])
    for cmd in commands:
        location = shutil.which(cmd)
        found = location is not None
        suffix = f" ({location})" if location else ""
        print(f"[{status(found):7}] command `{cmd}`{suffix}")
        all_ok = found and all_ok

    try:
        processes = int(os.environ.get("NPROC_PER_NODE", "1"))
        if processes < 1:
            raise ValueError
    except ValueError:
        print("[MISSING] NPROC_PER_NODE must be a positive integer")
        processes = 1
        all_ok = False
    if processes > 1:
        python = [conda_bin, "run", "-n", conda_env, "python3"] if conda_env else ["python3"]
        try:
            found = subprocess.run(
                python + ["-c", "import importlib.util; raise SystemExit(importlib.util.find_spec('torch.distributed.run') is None)"],
                capture_output=True, text=True, check=False,
            ).returncode == 0
        except OSError:
            found = False
        print(f"[{status(found):7}] module `torch.distributed.run` in the selected Python environment")
        all_ok = found and all_ok

    print("")
    for name in ["MODEL_PATH", "VILA_REPO", "CONDA_ENV"]:
        value = os.environ.get(name)
        print(f"[INFO   ] {name}={value if value is not None else '(not set)'}")

    if not all_ok:
        raise SystemExit("\nSetup check found missing required files or commands.")
    print("\nSetup check passed.")


if __name__ == "__main__":
    main()
