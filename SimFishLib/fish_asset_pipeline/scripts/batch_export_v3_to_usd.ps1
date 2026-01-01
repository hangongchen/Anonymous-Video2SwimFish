param(
  [string]$BlenderExe = "E:\blender\blender.exe",
  [string]$UsdPython = "E:\blender\5.1\python\bin\python.exe",
  [string]$ProjectRoot = "E:\Fish_dataset\fish_asset_pipeline",
  [string]$InputRoot = "E:\Fish_dataset\fish_asset_pipeline\generated_dataset\unlabeled_auto_skeleton_v3_spaced_bones",
  [string]$OutputRoot = "E:\Fish_dataset\fish_asset_pipeline\generated_dataset\usd_articulated_v1",
  [string]$Config = "E:\Fish_dataset\fish_asset_pipeline\configs\usd_physics_defaults.json"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutputRoot | Out-Null

$samples = Get-ChildItem -Path $InputRoot -Directory
foreach ($sample in $samples) {
  $blend = Join-Path $sample.FullName "auto_skeleton.blend"
  $skeletonJson = Join-Path $sample.FullName "skeleton.json"
  if (-not (Test-Path $blend) -or -not (Test-Path $skeletonJson)) {
    Write-Warning "Skipping $($sample.Name): missing auto_skeleton.blend or skeleton.json"
    continue
  }

  $safeName = $sample.Name -replace '[\\/:*?"<>|]', '_'
  $outDir = Join-Path $OutputRoot $safeName
  New-Item -ItemType Directory -Force $outDir | Out-Null
  $baseUsd = Join-Path $outDir "base_export.usda"
  $finalUsd = Join-Path $outDir "fish_articulated.usda"

  & $BlenderExe -b $blend --python (Join-Path $ProjectRoot "scripts\export_blend_to_usd.py") -- --output $baseUsd
  & $UsdPython (Join-Path $ProjectRoot "scripts\add_isaac_physics_to_usd.py") --input-usd $baseUsd --skeleton-json $skeletonJson --config $Config --output-usd $finalUsd
}
