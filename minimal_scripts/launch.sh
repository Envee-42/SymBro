#!/usr/bin/env bash
# minimal_scripts/launch.sh
#
# Copy this whole minimal_scripts/ folder into a new empty directory per
# project, edit the block below, run it. See Instructions.txt in this
# folder for why it stops after the broad `symbro geometry` pass instead
# of running the whole pipeline unattended.

set -euo pipefail

# ============================== EDIT THIS ==============================

# Search by symmetry + resolution (leave SYMMETRY="" to skip this filter
# entirely -- e.g. if you're only using ENTRY_ID below instead):
SYMMETRY="C3"
RESOLUTION_MAX="2.5"

# OR look up one specific entry directly by ID instead of searching --
# leave this empty to just use SYMMETRY/RESOLUTION_MAX above. Both can
# also be combined (ENTRY_ID narrows further within the search).
ENTRY_ID=""

# =========================================================================

QUERY_ARGS=()
if [ -n "$SYMMETRY" ]; then
    QUERY_ARGS+=(--symmetry "$SYMMETRY")
fi
if [ -n "$RESOLUTION_MAX" ]; then
    QUERY_ARGS+=(--resolution-max "$RESOLUTION_MAX")
fi
if [ -n "$ENTRY_ID" ]; then
    QUERY_ARGS+=(--entry-id "$ENTRY_ID")
fi

if [ ${#QUERY_ARGS[@]} -eq 0 ]; then
    echo "✗ Set at least one of SYMMETRY or ENTRY_ID at the top of this script." >&2
    exit 1
fi

echo "== symbro query ${QUERY_ARGS[*]} =="
symbro query "${QUERY_ARGS[@]}"

echo
echo "== symbro download =="
symbro download

echo
echo "== symbro geometry (broad pass -- see what's actually present) =="
symbro geometry

echo
echo "-----------------------------------------------------------------"
echo "Stopped here on purpose -- see Instructions.txt's 'WHY IT STOPS'."
echo "Pick a symmetry type from the table above, then continue by hand:"
echo "    symbro geometry --symmetry-type <e.g. C3>"
echo "    symbro isolate"
echo "    symbro rfdiffusion"
echo "    symbro pmpnn"
echo "    symbro predict"
echo "-----------------------------------------------------------------"
