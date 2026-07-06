# Build Cross Section Studio as a Windows onedir distribution (PyInstaller).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Installing runtime and build dependencies..."
python -m pip install -r requirements.txt -r requirements-build.txt -q

Write-Host "Running PyInstaller (this may take several minutes)..."
python -m PyInstaller cross_section_studio.spec --noconfirm --clean

$OutDir = Join-Path $Root "dist\CrossSectionStudio"
$Exe = Join-Path $OutDir "CrossSectionStudio.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "Build failed: $Exe not found"
}

$ZipPath = Join-Path $Root "dist\CrossSectionStudio-win64.zip"
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}
Write-Host "Creating zip archive..."
Compress-Archive -Path (Join-Path $OutDir "*") -DestinationPath $ZipPath -CompressionLevel Optimal

$ZipSizeMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)
Write-Host ""
Write-Host "Build complete."
Write-Host "  Folder: $OutDir"
Write-Host "  Run:    $Exe"
Write-Host "  Zip:    $ZipPath ($ZipSizeMb MB)"
