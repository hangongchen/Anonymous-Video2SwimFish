param(
  [string]$BlenderExe = "E:\blender\blender.exe",
  [string]$ProjectRoot = "E:\Fish_dataset\fish_asset_pipeline",
  [string]$UnlabeledDir = "E:\Fish_dataset\fish_asset_pipeline\dataset\dataset\unlabed_dataset\blender",
  [string]$OutputRoot = "E:\Fish_dataset\fish_asset_pipeline\generated_dataset\unlabeled_auto_skeleton_v3_spaced_bones",
  [string]$TemplateIndex = "E:\Fish_dataset\fish_asset_pipeline\configs\skeleton_template_index.json"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutputRoot | Out-Null

python (Join-Path $ProjectRoot "scripts\build_template_index.py") `
  --project-root $ProjectRoot `
  --output $TemplateIndex | Out-Null

$unlabeledFiles = Get-ChildItem -Path $UnlabeledDir -Filter "*.blend" -File |
  Where-Object { $_.Name -notlike "._*" }

foreach ($file in $unlabeledFiles) {
  $sampleName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
  $safeSampleName = $sampleName -replace '[\\/:*?"<>|]', '_'
  $outDir = Join-Path $OutputRoot $safeSampleName
  New-Item -ItemType Directory -Force $outDir | Out-Null

  & $BlenderExe -b $file.FullName --python (Join-Path $ProjectRoot "scripts\generate_auto_skeleton_blend.py") -- --template-index $TemplateIndex --output $outDir

  $generatedBlend = Join-Path $outDir "auto_skeleton.blend"
  if (Test-Path $generatedBlend) {
    & $BlenderExe -b $generatedBlend --python (Join-Path $ProjectRoot "scripts\extract_fish_skeleton.py") -- --output $outDir
  }
}
