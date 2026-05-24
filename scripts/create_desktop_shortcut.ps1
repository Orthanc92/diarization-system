$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExePath = Join-Path $Root "dist\DiarizationSystem.exe"

if (-not (Test-Path $ExePath)) {
    throw "Launcher exe was not found: $ExePath. Run scripts\setup_windows_app.ps1 first."
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Diarization System.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $ExePath
$Shortcut.WorkingDirectory = $Root
$Shortcut.IconLocation = "$ExePath,0"
$Shortcut.Description = "Start local Diarization System service"
$Shortcut.Save()

Write-Host "Shortcut created: $ShortcutPath"
