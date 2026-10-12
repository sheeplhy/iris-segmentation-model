$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python311 = Join-Path $env:LocalAppData "Programs\Python\Python311\python.exe"

if (-not (Test-Path -LiteralPath $python311)) {
    throw "Python 3.11 was not found at: $python311"
}

Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & $python311 -m venv .venv
}

$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
& $projectPython -m pip install --upgrade pip setuptools wheel
& $projectPython -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
& $projectPython -m pip install -e .
& $projectPython -m iris_system.healthcheck

Write-Host "Environment setup completed. Double-click start_gui.bat to launch the GUI." -ForegroundColor Green
