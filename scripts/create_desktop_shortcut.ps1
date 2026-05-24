$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExePath = Join-Path $Root "dist\DiarizationSystem.exe"
$StartBatPath = Join-Path $Root "start_windows.bat"

if (-not (Test-Path $ExePath) -and -not (Test-Path $StartBatPath)) {
    throw "Launcher was not found. Expected $ExePath or $StartBatPath."
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Diarization System.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.WorkingDirectory = $Root
$Shortcut.Description = "Start local Diarization System service"

if (Test-Path $ExePath) {
    $Shortcut.TargetPath = $ExePath
    $Shortcut.IconLocation = "$ExePath,0"
} else {
    $Shortcut.TargetPath = "$env:ComSpec"
    $Shortcut.Arguments = "/c `"$StartBatPath`""
    $Shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,220"
}

$Shortcut.Save()

Write-Host "Shortcut created: $ShortcutPath"
