$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 virtual environment creation failed.' }
    }
    & .venv/Scripts/python.exe -m pip install -r requirements.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & .venv/Scripts/python.exe -m pip install --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
    & .venv/Scripts/python.exe -m qwop_lab.cli bootstrap
    if ($LASTEXITCODE -ne 0) { throw 'Game/browser bootstrap failed.' }
} finally {
    Pop-Location
}
