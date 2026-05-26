param(
    [string]$ExeName = "DiarizationSystem.exe",
    [string]$ShortcutName = "Diarization System",
    [string]$Description = "Start local Diarization System service",
    [string]$Arguments = ""
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExePath = Join-Path $Root "dist\$ExeName"
$StartBatPath = Join-Path $Root "start_windows.bat"

if (-not (Test-Path $ExePath) -and -not (Test-Path $StartBatPath)) {
    throw "Launcher was not found. Expected $ExePath or $StartBatPath."
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "$ShortcutName.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.WorkingDirectory = $Root
$Shortcut.Description = $Description

if (Test-Path $ExePath) {
    $Shortcut.TargetPath = $ExePath
    $Shortcut.Arguments = $Arguments
    $Shortcut.IconLocation = "$ExePath,0"
} else {
    $Shortcut.TargetPath = "$env:ComSpec"
    $Shortcut.Arguments = "/c `"$StartBatPath`""
    $Shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,220"
}

$Shortcut.Save()

Write-Host "Shortcut created: $ShortcutPath"
