"""
config.py — loads per-machine installation settings (tool repo paths,
python executables, backend choice) out of ONE local YAML file, kept out
of git.

Why this exists
----------------
rfdiffusion.py and proteinmpnn.py both need to know things that are true
for YOUR machine and nobody else's: where your local RFdiffusion/
ProteinMPNN clone lives, which python/conda env has their dependencies,
whether to run RFdiffusion via a Singularity image. None of that belongs
hardcoded in the toolkit itself — it would break the moment anyone else
clones the repo, or you move to a different machine or cluster.

Simplicity is the whole design here: ONE file, installation.yaml, sitting
in your project root next to pyproject.toml — gitignored, same convention
prosculpt (a comparable published tool) uses for its own installation.yaml
— with one top-level key per tool:

    # installation.yaml
    rfdiffusion:
      backend: local                 # or "singularity"
      repo_path: /home/you/RFdiffusion
      python_executable: /home/you/miniconda3/envs/rfdiffusion/bin/python

    proteinmpnn:
      repo_path: /home/you/ProteinMPNN
      python_executable: /home/you/miniconda3/envs/proteinmpnn/bin/python

See installation.example.yaml (shipped alongside this file) for a
fill-in-the-blanks starting point — copy it to installation.yaml and edit
the two repo_path lines, that's the entire setup.

Usage
-----
    from toolkit import config, proteinmpnn

    cfg = config.load_installation_config()          # reads installation.yaml once
    sequences_df = proteinmpnn.run("results.zip", config=cfg)

Nothing here is required. Every submit()/run() function that accepts
config=... also accepts the same settings directly as keyword arguments
(repo_path=..., python_executable=...), and those always win over
whatever the file has. config=None (the default everywhere) just means
"use the function's own hardcoded fallback" — a fresh clone with no
installation.yaml yet still works, you just have to pass repo_path=...
by hand every call until you write the file.
"""

import os
from typing import Optional

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "config.py needs PyYAML to read installation.yaml — install it with "
        "`pip install pyyaml` (or add it to pyproject.toml/environment.yml)."
    ) from exc

# Name of the per-machine settings file. Lives in the project root (next
# to pyproject.toml) — NOT inside the toolkit package — and must be
# gitignored, since it's going to contain absolute paths specific to
# whoever's machine it's sitting on.
CONFIG_FILE_NAME = "installation.yaml"


def get_installation_config_path() -> str:
    """
    Absolute path to installation.yaml, resolved next to the current
    working directory — same "lives in your project root, not some
    hidden OS path" convention as download.py's temporary_files/ and
    colab.py's temporary_simulations/.
    """
    return os.path.join(os.getcwd(), CONFIG_FILE_NAME)


def load_installation_config(path: Optional[str] = None) -> dict:
    """
    Reads installation.yaml (or `path`, if given) into a plain dict —
    this is the `config` argument rfdiffusion.submit()/proteinmpnn.run()
    expect.

    No file yet -> {} (NOT an error). Every tool's own get_tool_config()
    call below already treats a missing section the same way, so a fresh
    clone with no installation.yaml at all keeps working exactly as
    before this file existed — you just pass repo_path=... explicitly
    until you write one.
    """
    path = path or get_installation_config_path()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def get_tool_config(config: Optional[dict], tool_name: str) -> dict:
    """
    Pulls out one tool's own section (e.g. config["proteinmpnn"]),
    defaulting to {} if that tool has no section yet (or config itself
    is None/empty) — so every downstream `tool_config.get("repo_path")`
    call just returns None instead of raising KeyError, and the caller's
    own default takes over from there.
    """
    return (config or {}).get(tool_name) or {}