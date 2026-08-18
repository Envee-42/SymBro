# 01_full_walkthrough/run.ps1
#
# Windows PowerShell equivalent of run.sh -- see README.md in this folder
# for the full narration and captured output this reproduces. Chains
# symbro's four free stages (no GPU, no installation.yaml needed) against
# real RCSB entry 8UF0.
#
# Does NOT run rfdiffusion/pmpnn/predict -- those need RFdiffusion/
# ProteinMPNN/a structure-prediction backend actually installed and
# configured first (see the main README). RFdiffusion itself has no
# native-Windows GPU path (needs WSL2) -- see the OS-compatibility table
# in the main README before assuming stage 5 runs here at all.

$ErrorActionPreference = "Stop"

$ProjectDir = "my_first_symbro_project"

if (Test-Path $ProjectDir) {
    Write-Error "$ProjectDir already exists -- remove it or edit `$ProjectDir in this script before rerunning."
    exit 1
}

New-Item -ItemType Directory -Path $ProjectDir | Out-Null
Set-Location $ProjectDir

Write-Host "== 1/5: symbro query --entry-id 8UF0 =="
symbro query --entry-id 8UF0

Write-Host "`n== 2/5: symbro download =="
symbro download

Write-Host "`n== 3/5: symbro geometry (broad pass) =="
symbro geometry

Write-Host "`n== 4/5: symbro geometry --symmetry-type C3 =="
symbro geometry --symmetry-type C3

Write-Host "`n== 5/5: symbro isolate =="
symbro isolate

Write-Host "`nDone. Ring structures ready for RFdiffusion are in $ProjectDir\temporary_subunits\."
Write-Host "`nNext (needs RFdiffusion/ProteinMPNN/a structure-prediction backend installed"
Write-Host "and configured -- see the main README's Installation/Configuration sections):"
Write-Host "    symbro rfdiffusion"
Write-Host "    symbro pmpnn"
Write-Host "    symbro predict"
