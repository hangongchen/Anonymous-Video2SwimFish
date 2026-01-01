param(
  [string]$BlenderExe = "blender",
  [string]$ProjectRoot = "E:\Fish_dataset\fish_asset_pipeline",
  [string]$SupervisedDir = "E:\Fish_dataset\fish_asset_pipeline\dataset\dataset\supervised_GT_dataset\fish_bone",
  [string]$UnlabeledDir = "E:\Fish_dataset\fish_asset_pipeline\dataset\dataset\unlabed_dataset\blender",
  [string]$OutputRoot = "E:\Fish_dataset\fish_asset_pipeline\generated_dataset\unlabeled_auto_skeleton"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutputRoot | Out-Null

$templates = Get-ChildItem -Path $SupervisedDir -Filter "*.blend" -File |
  Where-Object { $_.Name -notlike "._*" }

if ($templates.Count -eq 0) {
  throw "No supervised template blend files found."
}

function Normalize-FishName([string]$name) {
  $base = [System.IO.Path]::GetFileNameWithoutExtension($name).ToLowerInvariant()
  $base = $base -replace "_obj_\d+$", ""
  $base = $base -replace "_obj$", ""
  $base = $base -replace "_fish$", ""
  $base = $base -replace "\s+fish$", ""
  $base = $base -replace "[^a-z0-9]+", ""
  return $base
}

$templateByKey = @{}
foreach ($template in $templates) {
  $key = Normalize-FishName $template.Name
  if (-not $templateByKey.ContainsKey($key)) {
    $templateByKey[$key] = $template
  }
}

$fallbackTemplate = $templates[0]
$unlabeledFiles = Get-ChildItem -Path $UnlabeledDir -Filter "*.blend" -File |
  Where-Object { $_.Name -notlike "._*" }

foreach ($file in $unlabeledFiles) {
  $key = Normalize-FishName $file.Name
  $template = $fallbackTemplate
  $selectionReason = "fallback_first_supervised_template"
  if ($templateByKey.ContainsKey($key)) {
    $template = $templateByKey[$key]
    $selectionReason = "matched_by_normalized_file_name"
  }

  $sampleName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
  $safeSampleName = $sampleName -replace '[\\/:*?"<>|]', '_'
  $outDir = Join-Path $OutputRoot $safeSampleName
  New-Item -ItemType Directory -Force $outDir | Out-Null

  $selection = @{
    unlabeled_blend = $file.FullName
    template_blend = $template.FullName
    selection_reason = $selectionReason
  } | ConvertTo-Json -Depth 4
  Set-Content -Path (Join-Path $outDir "template_selection.json") -Value $selection -Encoding UTF8

  $templateSampleName = [System.IO.Path]::GetFileNameWithoutExtension($template.Name)
  $templateSkeletonJson = Join-Path $ProjectRoot (Join-Path "extracted_dataset\supervised" (Join-Path $templateSampleName "skeleton.json"))

  if (Test-Path $templateSkeletonJson) {
    & $BlenderExe -b $file.FullName --python (Join-Path $ProjectRoot "scripts\generate_auto_skeleton_blend.py") -- --template-blend $template.FullName --template-skeleton-json $templateSkeletonJson --output $outDir
  } else {
    & $BlenderExe -b $file.FullName --python (Join-Path $ProjectRoot "scripts\generate_auto_skeleton_blend.py") -- --template-blend $template.FullName --output $outDir
  }

  $generatedBlend = Join-Path $outDir "auto_skeleton.blend"
  if (Test-Path $generatedBlend) {
    & $BlenderExe -b $generatedBlend --python (Join-Path $ProjectRoot "scripts\extract_fish_skeleton.py") -- --output $outDir
  }
}
