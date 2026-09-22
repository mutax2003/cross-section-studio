# Build Cross Section Studio as a Windows onedir distribution (PyInstaller).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VersionFile = Join-Path $Root "VERSION"
if (-not (Test-Path $VersionFile)) {
    Write-Error "VERSION file not found at $VersionFile"
}
$Version = (Get-Content -Path $VersionFile -Raw).Trim().TrimStart("v", "V")
if (-not $Version) {
    Write-Error "VERSION file is empty"
}

Write-Host "Installing runtime and build dependencies..."
python -m pip install -r requirements.txt -r requirements-build.txt -q

Write-Host "Running PyInstaller for v$Version (this may take several minutes)..."
python -m PyInstaller cross_section_studio.spec --noconfirm --clean

$OutDir = Join-Path $Root "dist\CrossSectionStudio"
$Exe = Join-Path $OutDir "CrossSectionStudio.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "Build failed: $Exe not found"
}

$ZipName = "CrossSectionStudio-win64-v$Version.zip"
$ZipPath = Join-Path $Root "dist\$ZipName"
$LegacyZipPath = Join-Path $Root "dist\CrossSectionStudio-win64.zip"
foreach ($path in @($ZipPath, $LegacyZipPath)) {
    if (Test-Path $path) {
        Remove-Item $path -Force
    }
}

Write-Host "Creating zip archive..."
# Compress-Archive can hang / leave a 0-byte zip on large onedir trees; use .NET ZipFile.
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $OutDir,
    $ZipPath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)
Copy-Item -Path $ZipPath -Destination $LegacyZipPath -Force

$Sha256 = (Get-FileHash -Path $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
$ZipSizeMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)

$DefaultDownloadUrl = (
    "https://github.com/mutax2003/cross-section-studio/releases/download/" +
    "v$Version/$ZipName"
)
if ($env:CROSS_SECTION_RELEASE_DOWNLOAD_URL) {
    $DownloadUrl = $env:CROSS_SECTION_RELEASE_DOWNLOAD_URL.Trim()
} else {
    $DownloadUrl = $DefaultDownloadUrl
}

$Manifest = [ordered]@{
    version = $Version
    url     = $DownloadUrl
    sha256  = $Sha256
    notes   = "Cross Section Studio Windows desktop v$Version"
    zip     = $ZipName
}
$ManifestPath = Join-Path $Root "dist\release-manifest.json"
$Manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $ManifestPath -Encoding utf8

Write-Host ""
Write-Host "Build complete."
Write-Host "  Version: $Version"
Write-Host "  Folder:  $OutDir"
Write-Host "  Run:     $Exe"
Write-Host "  Zip:     $ZipPath ($ZipSizeMb MB)"
Write-Host "  Legacy:  $LegacyZipPath"
Write-Host "  SHA256:  $Sha256"
Write-Host "  Manifest:$ManifestPath"
