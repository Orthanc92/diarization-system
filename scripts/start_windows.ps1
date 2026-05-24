param(
    [switch]$NoBrowser,
    [switch]$SkipInstall,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsPath = Join-Path $Root "requirements.txt"
$RequirementsStamp = Join-Path $VenvDir ".requirements.sha256"
$LauncherPath = Join-Path $Root "launcher.py"
$PyTorchVersion = if ($env:PYTORCH_VERSION) { $env:PYTORCH_VERSION } else { "2.9.0" }
$TorchAudioVersion = if ($env:TORCHAUDIO_VERSION) { $env:TORCHAUDIO_VERSION } else { $PyTorchVersion }
$PyTorchIndexUrl = if ($env:PYTORCH_INDEX_URL) { $env:PYTORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cu128" }

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message =="
}

function Invoke-PythonCommand {
    param(
        [hashtable]$Python,
        [string[]]$Arguments
    )

    $allArgs = @()
    $allArgs += $Python.Args
    $allArgs += $Arguments
    & $Python.Command @allArgs
}

function Test-PythonCandidate {
    param(
        [string]$Command,
        [string[]]$Args
    )

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        return $null
    }

    $testArgs = @()
    $testArgs += $Args
    $testArgs += @(
        "-c",
        "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
    )

    & $Command @testArgs | Out-Null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    $versionArgs = @()
    $versionArgs += $Args
    $versionArgs += @(
        "-c",
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    )
    $version = (& $Command @versionArgs).Trim()

    return @{
        Command = $Command
        Args = $Args
        Version = $version
    }
}

function Get-SystemPython {
    $candidates = @(
        @{ Command = "py"; Args = @("-3") },
        @{ Command = "python"; Args = @() },
        @{ Command = "python3"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        $python = Test-PythonCandidate -Command $candidate.Command -Args $candidate.Args
        if ($python) {
            return $python
        }
    }

    return $null
}

function Ensure-Python {
    $python = Get-SystemPython
    if ($python) {
        Write-Host "Python found: $($python.Command) $($python.Args -join ' ') ($($python.Version))"
        return $python
    }

    Write-Host "Python 3.10+ was not found."
    Write-Host "Install Python from https://www.python.org/downloads/windows/ and enable 'Add python.exe to PATH'."
    try {
        Start-Process "https://www.python.org/downloads/windows/"
    } catch {
        Write-Host "Could not open browser automatically."
    }
    throw "Python 3.10+ is required."
}

function Ensure-Venv {
    param([hashtable]$Python)

    if (Test-Path $VenvPython) {
        Write-Host "Virtual environment found: $VenvDir"
        return
    }

    Write-Step "Creating virtual environment"
    Invoke-PythonCommand -Python $Python -Arguments @("-m", "venv", $VenvDir)
}

function Ensure-Dependencies {
    if (-not (Test-Path $RequirementsPath)) {
        throw "requirements.txt was not found: $RequirementsPath"
    }

    $requirementsHash = (Get-FileHash $RequirementsPath -Algorithm SHA256).Hash
    $installStamp = "requirements=$requirementsHash`ntorch=$PyTorchVersion`ntorchaudio=$TorchAudioVersion`nindex=$PyTorchIndexUrl"
    $installedHash = ""
    if (Test-Path $RequirementsStamp) {
        $installedHash = (Get-Content $RequirementsStamp -Raw).Trim()
    }

    if ($installStamp -eq $installedHash) {
        Write-Host "Dependencies are already installed for current requirements.txt."
        return
    }

    Write-Step "Installing dependencies"
    Write-Host "This can take a long time on the first run."
    Write-Host "Installing PyTorch $PyTorchVersion / torchaudio $TorchAudioVersion from $PyTorchIndexUrl"
    & $VenvPython -m pip install --upgrade pip

    & $VenvPython -m pip install `
        "torch==$PyTorchVersion" `
        "torchaudio==$TorchAudioVersion" `
        --index-url $PyTorchIndexUrl
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch installation failed."
    }

    $filteredRequirements = Join-Path $env:TEMP "diarization-system-requirements-no-torch.txt"
    Get-Content $RequirementsPath |
        Where-Object { $_ -notmatch "^\s*(torch|torchaudio|torchvision)\b" } |
        Set-Content -Path $filteredRequirements -Encoding UTF8

    & $VenvPython -m pip install -r $filteredRequirements
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }

    & $VenvPython -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'cuda', torch.version.cuda)"
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch verification failed."
    }

    Set-Content -Path $RequirementsStamp -Value $installStamp -Encoding ASCII
}

function Ensure-Shortcut {
    try {
        & (Join-Path $PSScriptRoot "create_desktop_shortcut.ps1")
    } catch {
        Write-Host "Desktop shortcut was not created: $($_.Exception.Message)"
    }
}

Set-Location $Root
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "uploads") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "output_files") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "browser_capture_chunks") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "model_cache") | Out-Null

Write-Step "Diarization System"
Write-Host "Project folder: $Root"

$python = Ensure-Python

if ($CheckOnly) {
    Write-Host "Check complete."
    Write-Host "Virtual environment exists: $(Test-Path $VenvPython)"
    exit 0
}

if (-not $SkipInstall) {
    Ensure-Venv -Python $python
    Ensure-Dependencies
    Ensure-Shortcut
} elseif (-not (Test-Path $VenvPython)) {
    throw "Virtual environment was not found and -SkipInstall was passed."
}

Write-Step "Starting service"
$launcherArgs = @($LauncherPath)
if ($NoBrowser) {
    $launcherArgs += "--no-browser"
}

& $VenvPython @launcherArgs
exit $LASTEXITCODE
