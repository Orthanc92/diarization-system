param(
    [switch]$NoShortcut
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $Root ".launcher-venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$DistExe = Join-Path $Root "dist\DiarizationSystemDocker.exe"

Set-Location $Root

function Get-SystemPython {
    $candidates = @(
        @{ Command = "py"; Args = @("-3") },
        @{ Command = "python"; Args = @() },
        @{ Command = "python3"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate.Command -ErrorAction SilentlyContinue)) {
            continue
        }
        $testArgs = @()
        $testArgs += $candidate.Args
        $testArgs += @(
            "-c",
            "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
        )
        & $candidate.Command @testArgs | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return $candidate
        }
    }

    return $null
}

$python = Get-SystemPython
if (-not $python) {
    throw "Python 3.10+ is required to build the launcher exe."
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating launcher virtual environment..."
    $venvArgs = @()
    $venvArgs += $python.Args
    $venvArgs += @("-m", "venv", $VenvDir)
    & $python.Command @venvArgs
}

Write-Host "Installing PyInstaller..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install pyinstaller

Write-Host "Building Docker launcher exe..."
& $VenvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name DiarizationSystemDocker `
    --distpath (Join-Path $Root "dist") `
    --workpath (Join-Path $Root "build\pyinstaller-docker") `
    --specpath (Join-Path $Root "build\pyinstaller-docker") `
    (Join-Path $Root "docker_launcher.py")

if (-not $NoShortcut) {
    Write-Host "Creating desktop shortcut..."
    & (Join-Path $PSScriptRoot "create_desktop_shortcut.ps1") `
        -ExeName "DiarizationSystemDocker.exe" `
        -ShortcutName "Diarization System Docker" `
        -Description "Start Diarization System Docker service"
}

Write-Host ""
Write-Host "Done."
Write-Host "Docker launcher: $DistExe"
Write-Host "Double-click it while Docker Desktop is running to start the service and open the browser."
