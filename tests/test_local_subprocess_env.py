"""
test_local_subprocess_env.py — the GLIBCXX-shadowing fix applied this
session to both local-execution backends (rfdiffusion._submit_local(),
pmpnn._run_local()): _conda_env_lib_dir()/_local_subprocess_env().
Parametrized across both modules since the logic is intentionally
duplicated (see each copy's own docstring for why).
"""
import os

import pytest

from toolkit import pmpnn, rfdiffusion

MODULES = [rfdiffusion, pmpnn]


@pytest.fixture
def fake_conda_env(tmp_path):
    env_root = tmp_path / "mambaforge-pypy3" / "envs" / "SE3nv"
    (env_root / "bin").mkdir(parents=True)
    (env_root / "lib").mkdir(parents=True)
    python_exe = env_root / "bin" / "python"
    python_exe.write_text("")
    return str(python_exe), str(env_root / "lib")


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_derives_lib_dir_from_env_layout(module, fake_conda_env):
    python_exe, lib_dir = fake_conda_env
    assert module._conda_env_lib_dir(python_exe) == lib_dir


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_bare_python_is_a_safe_no_op(module):
    assert module._conda_env_lib_dir("python") is None
    assert module._local_subprocess_env("python") is None


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_env_with_bin_but_no_lib_is_a_safe_no_op(module, tmp_path):
    bin_dir = tmp_path / "weird" / "bin"
    bin_dir.mkdir(parents=True)
    python_exe = bin_dir / "python"
    python_exe.write_text("")
    assert module._conda_env_lib_dir(str(python_exe)) is None


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_lib_dir_prepended_ahead_of_the_shadowing_system_path(module, fake_conda_env, monkeypatch):
    python_exe, lib_dir = fake_conda_env
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/ohpc/pub/compiler/gcc/9.3.0/lib64")

    env = module._local_subprocess_env(python_exe)
    assert env["LD_LIBRARY_PATH"] == f"{lib_dir}:/opt/ohpc/pub/compiler/gcc/9.3.0/lib64"
    assert env["LD_LIBRARY_PATH"].split(":")[0] == lib_dir  # env's own lib wins the race


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_handles_unset_ld_library_path(module, fake_conda_env, monkeypatch):
    python_exe, lib_dir = fake_conda_env
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    env = module._local_subprocess_env(python_exe)
    assert env["LD_LIBRARY_PATH"] == lib_dir  # no stray leading/trailing colon


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_rest_of_environment_is_preserved(module, fake_conda_env, monkeypatch):
    python_exe, _ = fake_conda_env
    monkeypatch.setenv("SOME_OTHER_VAR", "keep-me")
    env = module._local_subprocess_env(python_exe)
    assert env["SOME_OTHER_VAR"] == "keep-me"
    assert env["PATH"] == os.environ["PATH"]
