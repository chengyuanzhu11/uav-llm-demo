$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$flaskPidFile = Join-Path $projectRoot ".flask.pid"
$blocksRoot = "C:\Users\kyrie\AirSim\Cosys-AirSim-5.5-v3.3"

if (Test-Path -LiteralPath $flaskPidFile) {
    $flaskPid = Get-Content -LiteralPath $flaskPidFile -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($flaskPid -match "^\d+$") {
        $flaskProcess = Get-Process -Id ([int]$flaskPid) -ErrorAction SilentlyContinue
        if ($flaskProcess) {
            & taskkill.exe /PID $flaskProcess.Id /T /F | Out-Null
            Write-Output "Stopped Flask process tree $($flaskProcess.Id)."
        }
    }
    Remove-Item -LiteralPath $flaskPidFile -Force
}

Get-Process -Name "Blocks", "Blocks-Win64-Shipping" -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path.StartsWith($blocksRoot) } |
    ForEach-Object {
        Stop-Process -Id $_.Id -Force
        Write-Output "Stopped Cosys-AirSim process $($_.Id)."
    }
