$ErrorActionPreference = "Stop"

$blocksExe = "C:\Users\kyrie\AirSim\Cosys-AirSim-5.5-v3.3\Blocks_packaged_Windows_55_33\Windows\Blocks.exe"
$settings = Join-Path ([Environment]::GetFolderPath("MyDocuments")) "AirSim\settings.json"

if (-not (Test-Path -LiteralPath $blocksExe)) {
    throw "Cosys-AirSim Blocks executable was not found at $blocksExe"
}

if (-not (Test-Path -LiteralPath $settings)) {
    throw "AirSim settings file was not found at $settings"
}

$existing = Get-Process -Name "Blocks-Win64-Shipping", "Blocks" -ErrorAction SilentlyContinue
if (-not $existing) {
    Start-Process `
        -FilePath $blocksExe `
        -ArgumentList @(
            "-ResX=1280",
            "-ResY=720",
            "-windowed",
            "-dx11",
            "-settings=`"$settings`""
        ) `
        -WorkingDirectory (Split-Path -Parent $blocksExe)
}

$deadline = (Get-Date).AddSeconds(90)
do {
    Start-Sleep -Milliseconds 500
    $listener = Get-NetTCPConnection `
        -LocalPort 41451 `
        -State Listen `
        -ErrorAction SilentlyContinue
} while (-not $listener -and (Get-Date) -lt $deadline)

if (-not $listener) {
    throw "Blocks started, but AirSim RPC port 41451 did not open within 90 seconds."
}

Write-Output "Cosys-AirSim RPC is listening on 127.0.0.1:41451."
