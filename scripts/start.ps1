param([string]$Device = "", [int]$Port = 7860)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Create .venv and install requirements.txt first (see README.md)."
}
Push-Location (Join-Path $ProjectRoot "frontend")
try {
    if (-not (Test-Path -LiteralPath "node_modules")) {
        npm ci
        if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
    }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
} finally { Pop-Location }
Push-Location $ProjectRoot
try {
    $LaunchArgs = @("-m", "behavior_detection.cli", "app", "--port", "$Port")
    if ($Device) { $LaunchArgs += @("--device", $Device) }
    & $PythonExe @LaunchArgs
} finally { Pop-Location }
