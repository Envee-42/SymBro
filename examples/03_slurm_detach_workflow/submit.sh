#!/usr/bin/env bash
# 03_slurm_detach_workflow/submit.sh
#
# Thin wrapper around `symbro rfdiffusion --detach`, meant to be edited
# once per project and reused -- same idea as prosculpt's own
# Minimal_Prosculpt_scripts/launch.sh. Requires installation.yaml (in the
# project directory below) already configured with rfdiffusion.backend:
# slurm -- see README.md in this folder.
#
# Usage: ./submit.sh /path/to/your/project

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/your/project" >&2
    exit 1
fi

PROJECT_DIR="$1"

if [ ! -f "$PROJECT_DIR/installation.yaml" ]; then
    echo "✗ $PROJECT_DIR/installation.yaml not found -- copy installation.example.yaml there" >&2
    echo "  and set rfdiffusion.backend: slurm (plus its slurm: block) before running this." >&2
    exit 1
fi

cd "$PROJECT_DIR"

# Uncomment if you want this script to also activate a specific env on the
# submitting/login node itself (separate from installation.yaml's own
# setup_lines, which run on the compute node):
# source /path/to/your/conda.sh
# conda activate symbro

symbro rfdiffusion --detach

echo
echo "Submitted. Check progress later with: symbro status  (run from $PROJECT_DIR)"
