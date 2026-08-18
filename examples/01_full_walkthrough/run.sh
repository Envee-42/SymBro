#!/usr/bin/env bash
# 01_full_walkthrough/run.sh
#
# Chains symbro's four free stages (no GPU, no installation.yaml needed)
# against real RCSB entry 8UF0 -- see README.md in this folder for the
# full narration and captured output this script reproduces.
#
# Run from anywhere; it creates ./my_first_symbro_project itself.
#
# This does NOT run rfdiffusion/pmpnn/predict -- those need RFdiffusion/
# ProteinMPNN/a structure-prediction backend actually installed and
# configured first (see the main README). The commands for those three
# are printed at the end instead of run.

set -euo pipefail

PROJECT_DIR="my_first_symbro_project"

if [ -d "$PROJECT_DIR" ]; then
    echo "✗ $PROJECT_DIR already exists -- remove it or edit PROJECT_DIR in this script before rerunning." >&2
    exit 1
fi

mkdir "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "== 1/5: symbro query --entry-id 8UF0 =="
symbro query --entry-id 8UF0

echo
echo "== 2/5: symbro download =="
symbro download

echo
echo "== 3/5: symbro geometry (broad pass) =="
symbro geometry

echo
echo "== 4/5: symbro geometry --symmetry-type C3 =="
symbro geometry --symmetry-type C3

echo
echo "== 5/5: symbro isolate =="
symbro isolate

echo
echo "Done. Ring structures ready for RFdiffusion are in $PROJECT_DIR/temporary_subunits/."
echo
echo "Next (needs RFdiffusion/ProteinMPNN/a structure-prediction backend installed"
echo "and configured -- see the main README's Installation/Configuration sections):"
echo "    symbro rfdiffusion"
echo "    symbro pmpnn"
echo "    symbro predict"
