import json
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import requests

from build_qwen_train_data import iter_reva_qas


root = Path(__file__).resolve().parents[1]
url = "https://huggingface.co/datasets/ReVA-Benchmark/ReVA/resolve/main/"
for split in ("train", "test"):
    response = requests.get(url + f"{split}_set.json", timeout=120)
    response.raise_for_status()
    (root / f"data/reva_{split}/{split}_set.json").write_bytes(response.content)

train = json.loads((root / "data/reva_train/train_set.json").read_text())
test = json.loads((root / "data/reva_test/test_set.json").read_text())
paths = {item["file_path"] for item in islice(iter_reva_qas(train), 200)}
paths.update(item["file_path"] for item in test["videos"].values())
paths = sorted(path.removeprefix("#dataset/").removeprefix("ReVA_V2/") for path in paths)


def download(path):
    target = root / "data/ReVA" / path
    if target.is_file():
        return
    response = requests.get(url + path, timeout=120)
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_suffix(".part").write_bytes(response.content)
    target.with_suffix(".part").replace(target)


with ThreadPoolExecutor(max_workers=8) as pool:
    for count, _ in enumerate(pool.map(download, paths), 1):
        if count % 100 == 0:
            print(f"Videos: {count}/{len(paths)}", flush=True)

for group in {path.split("/")[0] for path in paths}:
    link = root / "data/qwen_train" / group
    if not link.exists():
        link.symlink_to(Path("../ReVA") / group, target_is_directory=True)
print(f"Ready: {len(paths)} videos for 200 training samples and 4,000 test questions.")
