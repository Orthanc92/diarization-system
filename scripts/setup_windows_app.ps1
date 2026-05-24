$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$DistExe = Join-Path $Root "dist\DiarizationSystem.exe"

Set-Location $Root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not found in PATH. Install Python 3.11/3.12 and run this script again."
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment..."
    python -m venv (Join-Path $Root ".venv")
}

Write-Host "Installing project dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")
& $VenvPython -m pip install pyinstaller

Write-Host "Building launcher exe..."
& $VenvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name DiarizationSystem `
    --distpath (Join-Path $Root "dist") `
    --workpath (Join-Path $Root "build\pyinstaller") `
    --specpath (Join-Path $Root "build\pyinstaller") `
    (Join-Path $Root "launcher.py")

Write-Host "Creating desktop shortcut..."
& (Join-Path $PSScriptRoot "create_desktop_shortcut.ps1")

Write-Host ""
Write-Host "Done."
Write-Host "Launcher: $DistExe"
Write-Host "Desktop shortcut: Diarization System"
Write-Host "Double-click the shortcut to start the service and open the browser."
