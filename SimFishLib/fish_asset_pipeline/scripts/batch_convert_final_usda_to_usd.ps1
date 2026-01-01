param(
  [string]$UsdPython = "E:\blender\5.1\python\bin\python.exe",
  [string]$ProjectRoot = "E:\Fish_dataset\fish_asset_pipeline",
  [string]$InputRoot = "E:\Fish_dataset\fish_asset_pipeline\generated_usd_dataset\usd_articulated_final_v1",
  [string]$OutputRoot = "E:\Fish_dataset\fish_asset_pipeline\generated_usd_dataset\usd_articulated_final_v1_binary"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutputRoot | Out-Null

$samples = Get-ChildItem -Path $InputRoot -Directory
foreach ($sample in $samples) {
  $inputUsd = Join-Path $sample.FullName "fish_articulated.usda"
  if (-not (Test-Path $inputUsd)) {
    Write-Warning "Skipping $($sample.Name): missing fish_articulated.usda"
    continue
  }

  $outDir = Join-Path $OutputRoot $sample.Name
  New-Item -ItemType Directory -Force $outDir | Out-Null
  $outputUsd = Join-Path $outDir "fish_articulated.usd"

  & $UsdPython (Join-Path $ProjectRoot "scripts\convert_usda_to_usd.py") --input $inputUsd --output $outputUsd
}
