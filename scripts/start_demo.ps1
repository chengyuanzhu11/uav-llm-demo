$ErrorActionPreference = "Stop"

function Test-PythonRuntime {
    param([string]$Candidate)

    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        return $false
    }

    try {
        & $Candidate -c "import flask" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$flaskPidFile = Join-Path $projectRoot ".flask.pid"
$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$bundledSitePackages = Join-Path $projectRoot ".venv\Lib\site-packages"
$pythonCandidates = @(
    $env:UAV_DEMO_PYTHON,
    "C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe",
    (Join-Path $projectRoot ".venv-airsim\Scripts\python.exe"),
    (Join-Path $projectRoot ".venv\Scripts\python.exe"),
    (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")
) | Where-Object { Test-PythonRuntime -Candidate $_ }
$python = $pythonCandidates | Select-Object -First 1

if (-not $python -and (Test-Path -LiteralPath $bundledPython -PathType Leaf) -and
    (Test-Path -LiteralPath $bundledSitePackages -PathType Container)) {
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = $bundledSitePackages
    if (Test-PythonRuntime -Candidate $bundledPython) {
        $python = $bundledPython
    } else {
        $env:PYTHONPATH = $previousPythonPath
    }
}

if (-not $python) {
    throw "No working Python with Flask was found. Set UAV_DEMO_PYTHON to a compatible python.exe."
}

$flaskListener = Get-NetTCPConnection `
    -LocalPort 5000 `
    -State Listen `
    -ErrorAction SilentlyContinue |
    Select-Object -First 1

$trackedPid = $null
if (Test-Path -LiteralPath $flaskPidFile) {
    $pidText = Get-Content -LiteralPath $flaskPidFile -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($pidText -match "^\d+$") {
        $trackedPid = [int]$pidText
    }
}

if ($flaskListener) {
    if ($trackedPid -and $flaskListener.OwningProcess -eq $trackedPid) {
        & taskkill.exe /PID $trackedPid /T /F | Out-Null
        $stopDeadline = (Get-Date).AddSeconds(10)
        do {
            Start-Sleep -Milliseconds 200
            $flaskListener = Get-NetTCPConnection `
                -LocalPort 5000 `
                -State Listen `
                -ErrorAction SilentlyContinue |
                Select-Object -First 1
        } while ($flaskListener -and (Get-Date) -lt $stopDeadline)

        if ($flaskListener) {
            throw "Tracked Flask process $trackedPid did not release port 5000."
        }
    } else {
        throw "Port 5000 is already in use by process $($flaskListener.OwningProcess); refusing to stop an untracked process."
    }
}

Remove-Item -LiteralPath $flaskPidFile -Force -ErrorAction SilentlyContinue

$flaskProcess = Start-Process `
    -FilePath $python `
    -ArgumentList @(
        "-m",
        "flask",
        "--app",
        "app",
        "run",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
        "--no-debugger",
        "--no-reload"
    ) `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $projectRoot "flask.out.log") `
    -RedirectStandardError (Join-Path $projectRoot "flask.err.log") `
    -PassThru
Set-Content -LiteralPath $flaskPidFile -Value $flaskProcess.Id

$deadline = (Get-Date).AddSeconds(30)
do {
    Start-Sleep -Milliseconds 300
    try {
        $state = Invoke-RestMethod `
            -Uri "http://127.0.0.1:5000/api/state" `
            -TimeoutSec 2
    } catch {
        $state = $null
    }
} while (-not $state -and (Get-Date) -lt $deadline)

if (-not $state) {
    throw "Flask did not become ready on http://127.0.0.1:5000."
}

Write-Output "UAV LLM demo is running with the HTML runtime."
Write-Output "Open http://127.0.0.1:5000"
Write-Output "Start Blocks separately before selecting AirSim Validation."
