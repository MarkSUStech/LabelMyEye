# Build LabelMyEye Windows installer
# Usage: powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not (Test-Path (Join-Path $Root "run.py"))) {
    $Root = Get-Location
}
Set-Location $Root

$PyInstaller = "$env:APPDATA\Python\Python310\Scripts\pyinstaller.exe"
if (-not (Test-Path $PyInstaller)) {
    $PyInstaller = (Get-Command pyinstaller -ErrorAction SilentlyContinue).Source
}
if (-not $PyInstaller) {
    python -m pip install pyinstaller --user
    $PyInstaller = "$env:APPDATA\Python\Python310\Scripts\pyinstaller.exe"
}

Write-Host "==> Building app with PyInstaller..." -ForegroundColor Cyan
& python -m PyInstaller --noconfirm --clean labelmyeye.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$exe = Join-Path $Root "dist\LabelMyEye\LabelMyEye.exe"
if (-not (Test-Path $exe)) { throw "Missing $exe" }

# Locate Inno Setup compiler
$ISCC = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 7\ISCC.exe",
    "$Root\tools\InnoSetup\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $ISCC) {
    $installer = Join-Path $Root "tools\innosetup-installer.exe"
    if (Test-Path $installer) {
        Write-Host "==> Installing Inno Setup (silent)..." -ForegroundColor Cyan
        $dest = Join-Path $Root "tools\InnoSetup"
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Start-Process -FilePath $installer -ArgumentList "/VERYSILENT","/SUPPRESSMSGBOXES","/NORESTART","/DIR=`"$dest`"" -Wait
        $ISCC = Join-Path $dest "ISCC.exe"
    }
}

if (-not (Test-Path $ISCC)) {
    throw "Inno Setup compiler (ISCC.exe) not found. Install Inno Setup 6+ and re-run."
}

Write-Host "==> Compiling installer with $ISCC ..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path (Join-Path $Root "dist\installer") | Out-Null
& $ISCC (Join-Path $Root "installer.iss")
if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }

$out = Get-ChildItem (Join-Path $Root "dist\installer\LabelMyEye-Setup-*.exe") | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host ""
Write-Host "Installer ready:" -ForegroundColor Green
Write-Host "  $($out.FullName)"
Write-Host "  Size: $([math]::Round($out.Length/1MB, 1)) MB"
