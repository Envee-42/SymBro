"""
config.py — loads the per-user "where are your tools actually installed"
settings (RFdiffusion's Singularity image path, GPU/bind-mount options,
and later ProteinMPNN/AF3/Boltz equivalents), following the same split
prosculpt's installation.yaml uses: a plain, gitignored YAML file holding
YOUR machine's real paths, copied from a checked-in template.

Deliberately plain YAML, not Hydra/OmegaConf: this project's own stated
goal is a simple, accessible workflow, and Hydra's config-composition
machinery is overkill for "here are a few file paths and flags."
RFdiffusion's own CLI still speaks Hydra key=value overrides underneath
(see rfdiffusion.py) — that's a property of the EXTERNAL tool, not
something this project's own config needs to inherit.
"""

import os

import yaml

DEFAULT_CONFIG_PATH = "installation.yaml"
_TEMPLATE_NAME = "installation.yaml.template"


def load_installation_config(path: str = None) -> dict:
    """
    Loads and returns the installation config as a plain dict, e.g.:

        rfdiffusion:
          backend: singularity
          singularity_image: /home/user/containers/rfdiffusion.sif
          singularity_executable: singularity   # or "apptainer" on some clusters
          use_gpu: true
          model_directory: /home/user/rfdiffusion_models   # optional
          bind_paths: []                        # extra host:container mounts
          python_executable: python              # only used for backend: local
          script_path: scripts/run_inference.py  # only used for backend: local

    Raises a clear, friendly error pointing at installation.yaml.template
    if `path` doesn't exist yet — the same "copy the template, fill it
    in" first-run experience prosculpt's own installation.yaml gives,
    rather than a bare FileNotFoundError with no next step.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No installation config found at {path!r}. Copy {_TEMPLATE_NAME} to "
            f"{path!r} (same folder) and fill in your machine's actual tool paths — "
            f"see that file's comments for what each field means. "
            f"{path!r} should stay out of version control (it's machine-specific)."
        )
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    return config


def get_tool_config(config: dict, tool_name: str) -> dict:
    """
    Returns config[tool_name] as a plain dict, or {} if that tool has no
    section yet — so callers can .get() sensible defaults freely instead
    of a KeyError for a tool the user hasn't configured yet (e.g. no
    ProteinMPNN section while only RFdiffusion is set up so far).
    """
    return dict(config.get(tool_name) or {})