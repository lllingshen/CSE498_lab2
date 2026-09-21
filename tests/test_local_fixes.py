import ast
import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text, expected", [
    ("Answer: (A) or (B)", ""),
    ("Do not choose A.", ""),
    ("Answer: **(A)** or **(B)**", ""),
    ("Answer: [A] / [B]", ""),
    ("(A) and (B)", ""),
    ("**A** or **B**", ""),
    ("Do not choose A. Final answer: B", "B"),
    ("Answer: (A) or (B). Final answer: C", "C"),
    ("<think>A and B are distractors.</think><answer>C</answer>", "C"),
    ("A and B are distractors. Final answer: C", "C"),
    ("The explanation matches A. Final answer: C", "C"),
    ("A", "A"),
    ("**(B)**", "B"),
    ("The answer is C because A and B are incorrect.", "C"),
])
def test_answer_parsers(text, expected):
    qwen = load_module("qwen_score", "scripts/score_reva_predictions.py")
    vila = load_module("vila_score", "vila_eval/reva_v2.py")
    choices = {"A": "a still image", "B": "a video clip", "C": "a camera"}
    assert qwen.extract_letter(text) == expected
    assert vila.parse_choice(text, choices) == (expected or None)


def test_option_text_matching():
    qwen = load_module("qwen_score", "scripts/score_reva_predictions.py")
    vila = load_module("vila_score", "vila_eval/reva_v2.py")
    choices = {"A": "a still image", "B": "a video clip", "C": "a camera"}
    for text in ["a video clip", "Answer: a video clip"]:
        scored, _ = qwen.score_predictions({"Q1": {"pred": text}}, [
            {"id": "Q1", "answer": "C", "options": choices}
        ])
        assert scored["Q1"]["pred_letter"] == "B"
        assert scored["Q1"]["acc"] == 0
        assert vila.parse_choice(text, choices) == "B"


def lora_functions():
    tree = ast.parse((ROOT / 'reva_eval/inference_vllm_origin_number.py').read_text())
    names = {'resolve_model_paths', 'load_lora_model', 'run_inference'}
    functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {'os': os, 'json': json, 'torch': SimpleNamespace(bfloat16='bf16'), 'ranki_print': Mock()}
    exec(compile(functions, '<loader tests>', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('weight_name', ['adapter_model.safetensors', 'adapter_model.bin'])
def test_lora_requires_config_and_weights(tmp_path, monkeypatch, weight_name):
    ns = lora_functions()
    resolve = ns['resolve_model_paths']
    with pytest.raises(ValueError, match='Invalid LoRA adapter directory'):
        resolve(str(tmp_path / 'missing'), 'base')
    config = tmp_path / 'adapter_config.json'
    config.write_text('{broken')
    with pytest.raises(ValueError, match='Invalid LoRA adapter directory'):
        resolve(str(tmp_path), 'base')
    config.write_text(json.dumps({'peft_type': 'LORA', 'base_model_name_or_path': 'base'}))
    for content in (None, b''):
        if content is not None:
            (tmp_path / weight_name).write_bytes(content)
        with pytest.raises(ValueError, match='missing nonempty'):
            resolve(str(tmp_path), 'base')
    (tmp_path / weight_name).write_bytes(b'test weight')
    peft_config = Mock(return_value=SimpleNamespace(base_model_name_or_path='base'))
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(PeftConfig=SimpleNamespace(from_pretrained=peft_config)))
    assert resolve(str(tmp_path)) == ('base', str(tmp_path))
    peft_config.side_effect = ValueError('bad config')
    with pytest.raises(ValueError, match='bad config'):
        resolve(str(tmp_path), 'base')


def test_lora_load_failure_does_not_fallback(tmp_path, monkeypatch):
    ns = lora_functions()
    (tmp_path / 'adapter_config.json').write_text('{"peft_type":"LORA"}')
    (tmp_path / 'adapter_model.bin').write_bytes(b'test weight')
    load_adapter = Mock(side_effect=RuntimeError('adapter mismatch'))
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(
        PeftConfig=SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(base_model_name_or_path='base'))),
        PeftModel=SimpleNamespace(from_pretrained=load_adapter),
    ))
    base_model = Mock()
    base_model.eval.return_value = base_model
    ns['load_transformers_vl_model'] = Mock(return_value=base_model)
    ns['auto_detect_world_size'] = Mock(side_effect=AssertionError('FSDP fallback reached'))
    args = SimpleNamespace(model_path=str(tmp_path), model_base='base', backend='transformers')
    with pytest.raises(RuntimeError, match='Failed to load LoRA adapter.*adapter mismatch'):
        ns['run_inference'](args)
    ns['auto_detect_world_size'].assert_not_called()
    assert not any('loaded successfully' in call.args[0] for call in ns['ranki_print'].call_args_list)
    adapted_model = Mock()
    adapted_model.eval.return_value = adapted_model
    load_adapter.side_effect = None
    load_adapter.return_value = adapted_model
    assert ns['load_lora_model']('base', str(tmp_path)) is adapted_model
    assert 'loaded successfully' in ns['ranki_print'].call_args.args[0]


@pytest.mark.parametrize('model_path', ['Qwen/Qwen3-VL-4B-Instruct', '/tmp/full-model-checkpoint'])
def test_full_model_path_is_preserved(model_path):
    ns = lora_functions()
    assert ns['resolve_model_paths'](model_path) == (model_path, None)
    model = Mock()
    model.eval.return_value = model
    ns['load_transformers_vl_model'] = Mock(return_value=model)
    ns['AutoProcessor'] = SimpleNamespace(from_pretrained=Mock(side_effect=RuntimeError('stop before inference')))
    args = SimpleNamespace(model_path=model_path, model_base=None, backend='transformers')
    with pytest.raises(RuntimeError, match='stop before inference'):
        ns['run_inference'](args)
    assert ns['load_transformers_vl_model'].call_args.args == (model_path,)
    assert ns['AutoProcessor'].from_pretrained.call_args.args == (model_path,)


def test_lora_shell_rejects_missing_adapter(tmp_path):
    env = {**os.environ, "MODEL_BASE": "base", "MODEL_PATH": str(tmp_path)}
    run = subprocess.run(["bash", str(ROOT / "scripts/run_eval_qwen_finetuned.sh")],
                         env=env, capture_output=True, text=True)
    assert run.returncode != 0
    assert "Invalid LoRA directory" in run.stderr
    assert "Preparing test set" not in run.stdout


def test_setup_overrides_and_venv(tmp_path, monkeypatch):
    m = load_module("setup_check", "scripts/check_student_setup.py")
    media = tmp_path / "media" / "VisDrone"
    media.mkdir(parents=True)
    (media / "clip.mp4").write_bytes(b"video")
    training = tmp_path / "training"
    training.mkdir()
    (training / "VisDrone").symlink_to(media, target_is_directory=True)
    qwen = training / "custom.json"
    qwen.write_text(json.dumps([{"video": "VisDrone/clip.mp4"}]))
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"videos": {"v": {"file_path": "#dataset/ReVA_V2/VisDrone/clip.mp4"}}}))
    for key, value in {"QWEN_TRAIN_JSON": qwen, "QWEN_VIDEO_ROOT": training,
                       "REVA_TRAIN_JSON": source, "REVA_JSON": source,
                       "REVA_ROOT": media.parent, "CONDA_ENV": "", "NPROC_PER_NODE": "1"}.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/python3" if name == "python3" else None)
    launcher = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(m.subprocess, "run", launcher)
    m.main()
    launcher.assert_not_called()
    monkeypatch.setenv("NPROC_PER_NODE", "2")
    m.main()
    assert "torch.distributed.run" in launcher.call_args.args[0][-1]
    launcher.return_value.returncode = 1
    with pytest.raises(SystemExit, match="missing required"):
        m.main()
    monkeypatch.setenv("NPROC_PER_NODE", "1")
    monkeypatch.setenv("CONDA_ENV", "qwen2")
    with pytest.raises(SystemExit, match="missing required"):
        m.main()


def test_setup_rejects_bad_references(tmp_path):
    m = load_module("setup_bad", "scripts/check_student_setup.py")
    annotation = tmp_path / "data.json"
    for data in ([{}], [{"video": ""}], [None], []):
        annotation.write_text(json.dumps(data))
        assert not m.check_qwen_video_references(annotation, tmp_path)
    for data in ({"videos": {"v": {}}}, {"videos": []}, {"videos": {}}):
        annotation.write_text(json.dumps(data))
        assert not m.check_reva_video_references(annotation, tmp_path)
    annotation.write_text("{")
    assert not m.check_qwen_video_references(annotation, tmp_path)
    assert not m.check_reva_video_references(annotation, tmp_path)
    (tmp_path / "empty.mp4").touch()
    assert m.resolve_video_reference(tmp_path, "empty.mp4") is None
    (tmp_path / "frames").mkdir()
    assert m.resolve_video_reference(tmp_path, "frames") is None
