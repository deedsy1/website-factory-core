param(
    [Parameter(Mandatory=$true)]
    [string]$ZipPath
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ZipPath)) {
    throw "Zip not found: $ZipPath"
}

$tempDir = Join-Path $env:TEMP ("patch_" + [System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tempDir | Out-Null

Expand-Archive -Path $ZipPath -DestinationPath $tempDir -Force

# Detect single top-level folder and flatten
$top = Get-ChildItem -Path $tempDir -Force
if ($top.Count -eq 1 -and $top[0].PSIsContainer) {
    $sourceRoot = $top[0].FullName
} else {
    $sourceRoot = $tempDir
}

# Copy from sourceRoot into current repo root
Get-ChildItem -Path $sourceRoot -Recurse -Force | ForEach-Object {
    $relativePath = $_.FullName.Substring($sourceRoot.Length).TrimStart("\","/")
    if ([string]::IsNullOrWhiteSpace($relativePath)) { return }

    $targetPath = Join-Path (Get-Location) $relativePath

    if ($_.PSIsContainer) {
        if (!(Test-Path $targetPath)) {
            New-Item -ItemType Directory -Path $targetPath | Out-Null
        }
    } else {
        $targetDir = Split-Path $targetPath
        if (!(Test-Path $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }
        Copy-Item -Path $_.FullName -Destination $targetPath -Force
    }
}

Remove-Item $tempDir -Recurse -Force
Write-Host "Patch applied successfully."
