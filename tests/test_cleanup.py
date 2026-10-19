import sys
from types import SimpleNamespace

import pytest

from scripts import cleanup


def blob(path, size=4096):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A fake project: one registered model, one stale export, HF cache, venv, junk, and source that must survive."""
    for rel in ["models/live-ov/m.bin", "models/old-experiment/m.bin", ".cache/hf/hub/ckpt.bin", ".cache/ov/1.blob", ".venv/lib/python3.12/site-packages/x.py",
                "server/__pycache__/a.pyc", ".euro.log", "server/app.py", "pyproject.toml", "uv.lock", ".git/HEAD", "README.md"]:
        blob(tmp_path / rel)
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    monkeypatch.setattr(cleanup, "REGISTRY", {"live": SimpleNamespace(path=tmp_path / "models/live-ov")})
    monkeypatch.setattr(cleanup, "server_running", lambda port=11434: False)
    monkeypatch.setattr(cleanup.shutil, "which", lambda name: None)  # never shell out to uv/nix during tests
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # keep ~/.cache/huggingface out of the picture
    monkeypatch.setattr(cleanup.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    return tmp_path


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["cleanup", *args])
    return cleanup.main()


SURVIVORS = ["server/app.py", "pyproject.toml", "uv.lock", ".git/HEAD", "README.md"]


def test_dry_run_deletes_nothing(tree, monkeypatch, capsys):
    assert run(monkeypatch, "--everything") == 0
    assert "dry run" in capsys.readouterr().out
    assert all((tree / p).exists() for p in SURVIVORS + ["models/live-ov/m.bin", ".cache/hf", ".venv", ".euro.log"])


def test_all_keeps_what_is_needed_to_serve(tree, monkeypatch):
    assert run(monkeypatch, "--all", "--yes") == 0
    assert not (tree / "models/old-experiment").exists()
    assert not (tree / ".cache/hf").exists() and not (tree / ".euro.log").exists() and not (tree / "server/__pycache__").exists()
    assert (tree / "models/live-ov/m.bin").exists() and (tree / ".venv").exists()
    assert (tree / ".cache/ov/1.blob").exists()  # compile cache is a speed-up for serving: not in --all
    assert all((tree / p).exists() for p in SURVIVORS)


def test_everything_removes_venv_and_models_but_never_source(tree, monkeypatch):
    assert run(monkeypatch, "--everything", "--yes") == 0
    assert not (tree / ".cache/ov").exists()
    assert not (tree / ".venv").exists() and not (tree / "models/live-ov").exists() and not (tree / ".cache/hf").exists()
    assert all((tree / p).exists() for p in SURVIVORS)


def test_single_step(tree, monkeypatch):
    assert run(monkeypatch, "--conversion-cache", "--yes") == 0
    assert not (tree / ".cache/hf").exists() and (tree / "models/old-experiment").exists()


def test_refuses_while_server_runs(tree, monkeypatch):
    monkeypatch.setattr(cleanup, "server_running", lambda port=11434: True)
    assert run(monkeypatch, "--all", "--yes") == 1
    assert (tree / ".cache/hf").exists()
    assert run(monkeypatch, "--all", "--yes", "--force") == 0


def test_remove_refuses_outside_project_and_protected(tree, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "keep"
    blob(outside)
    for bad in (outside, tree, tree / "server", tree / ".git", tree / "uv.lock"):
        with pytest.raises(RuntimeError):
            cleanup.remove(bad)
    assert outside.exists() and (tree / "server/app.py").exists()


def test_symlink_is_unlinked_not_followed(tree):
    target = tree / "models/live-ov"
    link = tree / ".cache/link"
    link.symlink_to(target)
    cleanup.remove(link)
    assert not link.exists() and (target / "m.bin").exists()
