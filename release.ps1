$env:TWINE_USERNAME = $env:TWINE_DSALT_USERNAME
$env:TWINE_PASSWORD = $env:TWINE_DSALT_PASSWORD
$ErrorActionPreference = "Stop" 

if (-not (Test-Path "pyproject.toml")) {
    Write-Host "ERROR: pyproject.toml not found!" -ForegroundColor Red
    exit 1
}

if (-not $env:TWINE_USERNAME -or -not $env:TWINE_PASSWORD) {
    Write-Host "ERROR: TWINE_DSALT_USERNAME / TWINE_DSALT_PASSWORD not set!" -ForegroundColor Red
    exit 1
}

$content = Get-Content "pyproject.toml" -Raw
$verMatch = [regex]::Match($content, '(?m)^\s*version\s*=\s*"(?<ver>[^"]*)"')
if ($verMatch.Success) {
    $currentVersion = $verMatch.Groups['ver'].Value
    Write-Host "Current version: $currentVersion" -ForegroundColor Cyan
} else {
    Write-Host "ERROR: Could not read the version from pyproject.toml!" -ForegroundColor Red
    exit 1
}

$newVersion = Read-Host "Enter new version"
if (-not $newVersion -or $newVersion -eq $currentVersion) {
    Write-Host "Invalid or identical version!" -ForegroundColor Yellow
    exit 1
}
if ($newVersion -notmatch '^\d+\.\d+\.\d+([.-][0-9A-Za-z.]+)?$') {
    Write-Host "ERROR: '$newVersion' is not a valid version (expected X.Y.Z)!" -ForegroundColor Red
    exit 1
}

$confirm = Read-Host "Publish v$newVersion to PyPI? This operation is IRREVERSIBLE. [y/N]"
if ($confirm -ne "y") {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

try {
    $newContent = [regex]::Replace(
        $content,
        '(?m)^(\s*version\s*=\s*")[^"]*(")',
        "`${1}$newVersion`${2}"
    )
    [System.IO.File]::WriteAllText("$(Get-Location)\pyproject.toml", $newContent)
} catch {
    Write-Host "ERROR: Failed to write pyproject.toml!" -ForegroundColor Red
    exit 1
}

if (Test-Path "dist") { Remove-Item "dist\*" -Force -Recurse -ErrorAction SilentlyContinue }

python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed! Restoring version..." -ForegroundColor Red
    [System.IO.File]::WriteAllText("$(Get-Location)\pyproject.toml", $content)
    exit 1
}

$f = Get-ChildItem -Path "dist" -Filter "*$newVersion*" | Select-Object -ExpandProperty FullName
python -m twine upload $f
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyPI upload failed! Restoring version..." -ForegroundColor Red
    [System.IO.File]::WriteAllText("$(Get-Location)\pyproject.toml", $content)
    exit 1
}

git add .
git commit -m "v$newVersion"
git push
Write-Host "--- RELEASE v$newVersion COMPLETED ---" -ForegroundColor Green
