param(
  [Parameter(Mandatory = $true)]
  [string]$InputBlend,

  [Parameter(Mandatory = $true)]
  [string]$OutputDir,

  [string]$BlenderExe = "E:\blender\blender.exe",
  [string]$UsdPython = "E:\blender\5.1\python\bin\python.exe",
  [string]$ProjectRoot = "E:\Fish_dataset\fish_asset_pipeline",
  [string]$TemplateIndex = "E:\Fish_dataset\fish_asset_pipeline\configs\skeleton_template_index.json",
  [string]$Config = "E:\Fish_dataset\fish_asset_pipeline\configs\usd_physics_defaults.json",
  [double]$TargetLengthM = 0.5,
  [switch]$KeepIntermediate
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $InputBlend)) {
  throw "Input blend file does not exist: $InputBlend"
}
if (-not (Test-Path $BlenderExe)) {
  throw "Blender executable does not exist: $BlenderExe"
}
if (-not (Test-Path $UsdPython)) {
  throw "USD Python executable does not exist: $UsdPython"
}
if (-not (Test-Path $ProjectRoot)) {
  throw "Project root does not exist: $ProjectRoot"
}
if (-not (Test-Path $Config)) {
  throw "Physics config does not exist: $Config"
}
if ($TargetLengthM -le 0) {
  throw "TargetLengthM must be positive. Got: $TargetLengthM"
}

New-Item -ItemType Directory -Force $OutputDir | Out-Null

if (-not (Test-Path $TemplateIndex)) {
  & $UsdPython (Join-Path $ProjectRoot "scripts\build_template_index.py") `
    --project-root $ProjectRoot `
    --output $TemplateIndex | Out-Null
}

$workDir = Join-Path $OutputDir "_work"
$skeletonDir = Join-Path $workDir "auto_skeleton"
$usdWorkDir = Join-Path $workDir "usd"
New-Item -ItemType Directory -Force $skeletonDir, $usdWorkDir | Out-Null

$autoBlend = Join-Path $skeletonDir "auto_skeleton.blend"
$skeletonJson = Join-Path $skeletonDir "skeleton.json"
$baseUsd = Join-Path $usdWorkDir "base_export.usda"
$finalUsda = Join-Path $OutputDir "fish_articulated.usda"
$finalUsd = Join-Path $OutputDir "fish_articulated.usd"

function Require-File {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$Step
  )
  if (-not (Test-Path $Path)) {
    throw "$Step failed. Missing expected file: $Path"
  }
}

Write-Host "[1/5] Generate real-size auto skeleton blend..."
& $BlenderExe -b $InputBlend --python (Join-Path $ProjectRoot "scripts\generate_auto_skeleton_blend.py") -- --template-index $TemplateIndex --target-length-m $TargetLengthM --output $skeletonDir
Require-File -Path $autoBlend -Step "Auto skeleton generation"
Write-Host "[2/5] Extract skeleton.json..."
& $BlenderExe -b $autoBlend --python (Join-Path $ProjectRoot "scripts\extract_fish_skeleton.py") -- --output $skeletonDir
Require-File -Path $skeletonJson -Step "Skeleton extraction"
Write-Host "[3/5] Export base USDA..."
& $BlenderExe -b $autoBlend --python (Join-Path $ProjectRoot "scripts\export_blend_to_usd.py") -- --output $baseUsd
Require-File -Path $baseUsd -Step "Base USD export"
Write-Host "[4/5] Add Isaac Sim physics, joints, attachments, mass, materials, and ground plane..."
& $UsdPython (Join-Path $ProjectRoot "scripts\add_isaac_physics_to_usd.py") --input-usd $baseUsd --skeleton-json $skeletonJson --config $Config --output-usd $finalUsda
Require-File -Path $finalUsda -Step "Isaac physics USD generation"
Write-Host "[5/5] Convert USDA to binary USD..."
& $UsdPython (Join-Path $ProjectRoot "scripts\convert_usda_to_usd.py") --input $finalUsda --output $finalUsd
Require-File -Path $finalUsd -Step "Binary USD conversion"

if (-not $KeepIntermediate) {
  Remove-Item -Recurse -Force $workDir -ErrorAction SilentlyContinue
}

Write-Host "Generated readable USDA: $finalUsda"
Write-Host "Generated USD: $finalUsd"
