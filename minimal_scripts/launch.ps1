# minimal_scripts/launch.ps1
#
# Windows PowerShell equivalent of launch.sh. Copy this whole
# minimal_scripts\ folder into a new empty directory per project, edit
# the block below, run it. See Instructions.txt in this folder for why
# it stops after the broad `symbro geometry` pass instead of running the
# whole pipeline unattended.

$ErrorActionPreference = "Stop"

# ============================== EDIT THIS ===============================

# Search by symmetry + resolution (leave $Symmetry = "" to skip this
# filter entirely -- e.g. if you're only using $EntryId below instead):
$Symmetry = "C3"
$ResolutionMax = "2.5"

# OR look up one specific entry directly by ID instead of searching --
# leave this empty to just use $Symmetry/$ResolutionMax above. Both can
# also be combined ($EntryId narrows further within the search).
$EntryId = ""

# ==========================================================================

$QueryArgs = @()
if ($Symmetry) { $QueryArgs += @("--symmetry", $Symmetry) }
if ($ResolutionMax) { $QueryArgs += @("--resolution-max", $ResolutionMax) }
if ($EntryId) { $QueryArgs += @("--entry-id", $EntryId) }

if ($QueryArgs.Count -eq 0) {
    Write-Error "Set at least one of `$Symmetry or `$EntryId at the top of this script."
    exit 1
}

Write-Host "== symbro query $QueryArgs =="
symbro query @QueryArgs

Write-Host "`n== symbro download =="
symbro download

Write-Host "`n== symbro geometry (broad pass -- see what's actually present) =="
symbro geometry

Write-Host "`n-----------------------------------------------------------------"
Write-Host "Stopped here on purpose -- see Instructions.txt's 'WHY IT STOPS'."
Write-Host "Pick a symmetry type from the table above, then continue by hand:"
Write-Host "    symbro geometry --symmetry-type <e.g. C3>"
Write-Host "    symbro isolate"
Write-Host "    symbro rfdiffusion"
Write-Host "    symbro pmpnn"
Write-Host "    symbro predict"
Write-Host "-----------------------------------------------------------------"
