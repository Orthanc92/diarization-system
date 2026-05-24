$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$DistExe = Join-Path $Root "dist\DiarizationSystem.exe"
$PyTorchVersion = if ($env:PYTORCH_VERSION) { $env:PYTORCH_VERSION } else { "2.9.0" }
$TorchAudioVersion = if ($env:TORCHAUDIO_VERSION) { $env:TORCHAUDIO_VERSION } else { $PyTorchVersion }
$PyTorchIndexUrl = if ($env:PYTORCH_INDEX_URL) { $env:PYTORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cu128" }

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
& $VenvPython -m pip install "torch==$PyTorchVersion" "torchaudio==$TorchAudioVersion" --index-url $PyTorchIndexUrl

$filteredRequirements = Join-Path $env:TEMP "diarization-system-requirements-no-torch.txt"
Get-Content (Join-Path $Root "requirements.txt") |
    Where-Object { $_ -notmatch "^\s*(torch|torchaudio|torchvision)\b" } |
    Set-Content -Path $filteredRequirements -Encoding UTF8
& $VenvPython -m pip install -r $filteredRequirements
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
