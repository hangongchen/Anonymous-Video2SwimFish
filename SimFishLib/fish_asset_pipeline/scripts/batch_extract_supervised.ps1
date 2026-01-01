param(
  [string]$BlenderExe = "blender",
  [string]$ProjectRoot = "E:\Fish_dataset\fish_asset_pipeline",
  [string]$SupervisedDir = "E:\Fish_dataset\fish_asset_pipeline\dataset\dataset\supervised_GT_dataset\fish_bone",
  [string]$OutputRoot = "E:\Fish_dataset\fish_asset_pipeline\extracted_dataset\supervised"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutputRoot | Out-Null

$files = Get-ChildItem -Path $SupervisedDir -Filter "*.blend" -File |
  Where-Object { $_.Name -notlike "._*" }

foreach ($file in $files) {
  $sampleName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
  $outDir = Join-Path $OutputRoot $sampleName
  New-Item -ItemType Directory -Force $outDir | Out-Null
  & $BlenderExe -b $file.FullName --python (Join-Path $ProjectRoot "scripts\extract_fish_skeleton.py") -- --output $outDir
}
