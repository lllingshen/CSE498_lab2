import argparse
import copy
import json
import os
import re
from pathlib import Path
from time import strftime
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=".data/ReVA_V2/valid_set.json")
    parser.add_argument("--dataset-root", type=str, default=".data")
    parser.add_argument("--dataset-prefix", type=str, default="#dataset")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--video-max-tiles", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timestamp-output", action="store_true")
    return parser.parse_args()


def load_instances(question_file: str) -> list[dict[str, Any]]:
    """Flatten nested ReVA annotations into VILA evaluation instances."""
    data = json.loads(Path(question_file).read_text(encoding="utf-8"))
    instances = []
    for video_id, video in data.get("videos", {}).items():
        video_path = video.get("file_path", "")
        for category, subcategories in video.get("mcq", {}).items():
            for subcategory, questions in subcategories.items():
                for index, qa in enumerate(questions):
                    qa_id = qa.get("qa_id")
                    if qa_id is None or qa_id == "":
                        if qa.get("global_index") is not None:
                            qa_id = f"REVA-G{int(qa['global_index']):06d}"
                        else:
                            stem = Path(video_path.replace("\\", "/")).stem
                            qa_id = f"REVA-{stem}-{subcategory[:3].upper()}-{index:04d}"
                    instances.append({**qa, "qa_id": str(qa_id), "video_id": video_id,
                                      "video_path": video_path, "dataset_name": video.get("dataset_name"),
                                      "category": category, "subcategory": subcategory,
                                      "correct_answer": str(qa.get("correct_answer", "")).strip().upper()})
    return instances


def resolve_video_path(raw_path: str, dataset_root: str, dataset_prefix: str) -> str:
    """Resolve a ReVA annotation path to an existing local video path."""
    raw_path = raw_path.replace("\\", "/")
    root = Path(dataset_root).expanduser()
    relative = raw_path
    prefix = dataset_prefix.rstrip("/") + "/"
    if relative.startswith(prefix):
        relative = relative[len(prefix):]
    candidates = [Path(raw_path).expanduser(), root / raw_path.lstrip("/"), root / relative.lstrip("/")]
    if relative.startswith("ReVA_V2/"):
        candidates.append(root / relative[len("ReVA_V2/"):])
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    raise FileNotFoundError(f"Cannot resolve video {raw_path!r} under {dataset_root!r}")


def build_prompt(question: str, options: dict[str, str]) -> str:
    """Build the VILA multiple-choice prompt."""
    lines = [question.strip()]
    lines.extend(f"{key}. {options[key]}" for key in sorted(options))
    lines.append("Answer with only the option letter from the given choices.")
    return "\n".join(lines)


def parse_choice(response: str, options: dict[str, str]) -> str | None:
    """Parse a choice letter from VILA raw response."""
    text = str(response or "")
    answers = re.findall(r"<answer\b[^>]*>(.*?)</answer\s*>", text, re.I | re.S)
    text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", answers[-1] if answers else text, flags=re.I | re.S).strip()
    choices = {str(key).upper(): str(value) for key, value in options.items()}
    for prefix in (r"final\s+answer", r"answer|correct\s+(?:choice|option)|(?:choose|select|pick)"):
        matches = list(re.finditer(
            rf"\b(?:{prefix})\s*(?:is\s*|:\s*|=\s*)?(?:option\s*)?[\s(*`\[\"']*([A-H])\b",
            text, re.I,
        ))
        if matches:
            match = matches[-1]
            if re.match(r"\s*(?:or|/|and)\s*[([]?[A-H]\b", text[match.end():], re.I):
                return None
            if match.group(1) == "a" and re.match(r"\s+[a-z]", text[match.end():]):
                continue
            letter = match.group(1).upper()
            return letter if letter in choices else None
    match = re.fullmatch(r"(?:option\s+)?[\s\W]*([A-H])[\s\W]*", text, re.I)
    if not match:
        match = re.match(r"^\s*(?:option\s+)?[(*`\[]*([A-H])[.)\]:-](?:\s|$)", text, re.I)
    if match:
        letter = match.group(1).upper()
        return letter if letter in choices else None
    text = re.split(r"\b(?:final\s+answer|answer)\s*(?::|=|\bis\b)\s*", text, flags=re.I)[-1]
    normalized = re.sub(r"\W+", " ", text.casefold()).strip()
    normalized = re.sub(r"^(?:(?:it|this) is|(?:the )?(?:correct )?(?:answer|choice|option) is)\s+", "", normalized)
    normalized = re.sub(r"^(?:a|an|the)\s+", "", normalized)
    matches = []
    for letter, value in choices.items():
        value = re.sub(r"\W+", " ", value.casefold()).strip()
        value = re.sub(r"^(?:a|an|the)\s+", "", value)
        if value and normalized == value:
            matches.append(letter)
    return matches[0] if len(matches) == 1 else None


def load_existing_predictions(output_path: str) -> dict[str, dict]:
    if not os.path.exists(output_path):
        return {}
    predictions = {}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predictions[record["qa_id"]] = record
    return predictions


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict]) -> dict:
    """Compute total, category, and subcategory accuracy for VILA outputs."""
    metrics = {"num_questions": 0, "num_answered": 0, "num_correct": 0, "accuracy": 0.0,
               "by_category": {}, "by_subcategory": {}}
    for record in records:
        answered = bool(re.fullmatch(r"[A-H]", str(record.get("pred_letter") or "").strip().upper()))
        correct = answered and record.get("is_correct") is True
        groups = [metrics]
        for field, output in (("category", "by_category"), ("subcategory", "by_subcategory")):
            groups.append(metrics[output].setdefault(record.get(field, "unknown"),
                          {"num_questions": 0, "num_answered": 0, "num_correct": 0, "accuracy": 0.0}))
        for group in groups:
            group["num_questions"] += 1
            group["num_answered"] += answered
            group["num_correct"] += correct
            group["accuracy"] = group["num_correct"] / group["num_questions"]
    return metrics


def main() -> None:
    args = parse_args()

    from tqdm import tqdm

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import llava
    from llava import Video
    from llava import conversation as conversation_lib

    instances = load_instances(args.question_file)
    if args.max_questions is not None:
        instances = instances[: args.max_questions]

    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir if args.resume or not args.timestamp_output else base_output_dir / strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation artifacts to: {output_dir}")

    outputs_path = output_dir / "outputs.jsonl"
    metrics_path = output_dir / "metrics.json"
    existing = load_existing_predictions(str(outputs_path)) if args.resume else {}

    if args.conv_mode != "auto":
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    model = llava.load(args.model_path, model_base=args.model_base)
    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames
    if args.video_max_tiles > 0:
        model.config.video_max_tiles = args.video_max_tiles
        model.llm.config.video_max_tiles = args.video_max_tiles

    generation_config = copy.deepcopy(model.default_generation_config)
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    outputs = []
    for instance in tqdm(instances):
        if instance["qa_id"] in existing:
            outputs.append(existing[instance["qa_id"]])
            continue

        video_path = resolve_video_path(instance["video_path"], args.dataset_root, args.dataset_prefix)
        prompt = build_prompt(instance["question"], instance["options"])
        response = model.generate_content([Video(video_path), prompt], generation_config=generation_config)
        pred_letter = parse_choice(response, instance["options"])

        outputs.append({
            "qa_id": instance["qa_id"],
            "video_id": instance["video_id"],
            "video_path": video_path,
            "dataset_name": instance.get("dataset_name"),
            "category": instance["category"],
            "subcategory": instance["subcategory"],
            "question": instance["question"],
            "options": instance["options"],
            "prompt": prompt,
            "raw_response": response,
            "pred_letter": pred_letter,
            "correct_answer": instance["correct_answer"],
            "is_correct": pred_letter == instance["correct_answer"],
            "reasoning": instance.get("reasoning", ""),
            "example": instance.get("example", ""),
        })
        save_jsonl(str(outputs_path), outputs)

    save_jsonl(str(outputs_path), outputs)
    save_json(str(metrics_path), summarize(outputs))


if __name__ == "__main__":
    main()
