param(
    [string]$BlocksExe = (Join-Path $env:USERPROFILE "AirSim\Cosys-AirSim-5.5-v3.3\Blocks_packaged_Windows_55_33\Windows\Blocks.exe"),
    [string]$Settings = (Join-Path ([Environment]::GetFolderPath("MyDocuments")) "AirSim\settings.json"),
    [string]$HostAddress,
    [switch]$DryRun,
    [switch]$FirewallOnly
)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

if (-not (Test-Path -LiteralPath $BlocksExe -PathType Leaf)) { throw "Blocks executable not found: $BlocksExe" }
if (-not (Test-Path -LiteralPath $Settings -PathType Leaf)) { throw "AirSim settings not found: $Settings" }
$simulatorExe = Join-Path (Split-Path -Parent $BlocksExe) "Blocks\Binaries\Win64\Blocks.exe"
if (-not (Test-Path -LiteralPath $simulatorExe -PathType Leaf)) { throw "Blocks simulator executable not found: $simulatorExe" }
$adapter = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -eq "vEthernet (WSL)" } | Select-Object -First 1
if (-not $HostAddress) {
    if (-not $adapter) { throw "Start WSL first or supply -HostAddress 127.0.0.1 for mirrored networking." }
    $HostAddress = $adapter.IPAddress
}
if ($HostAddress -ne "127.0.0.1" -and (-not $adapter -or $HostAddress -ne $adapter.IPAddress)) {
    throw "Only Windows localhost or the WSL virtual adapter address is allowed."
}
$config = Get-Content -LiteralPath $Settings -Raw | ConvertFrom-Json
if ($config.SimMode -ne "Multirotor" -or @($config.Vehicles.PSObject.Properties).Count -eq 0) {
    throw "This launcher requires a Multirotor SimpleFlight simulation."
}
foreach ($vehicle in $config.Vehicles.PSObject.Properties) {
    if ($vehicle.Value.VehicleType -ne "SimpleFlight") { throw "Only simulated SimpleFlight vehicles are allowed by this launcher." }
}
$config | Add-Member -NotePropertyName LocalHostIp -NotePropertyValue $HostAddress -Force
$port = if ($config.ApiServerPort) { [int]$config.ApiServerPort } else { 41451 }
$cache = Join-Path $env:LOCALAPPDATA "UAV_LLM\AirSim"
$launchSettings = Join-Path $cache "settings-wsl.json"
$ruleName = "UAV-LLM-WSL-AirSim"
if ($DryRun) {
    [pscustomobject]@{BlocksExe=$BlocksExe; SimulatorExe=$simulatorExe; OriginalSettings=$Settings;
        LaunchSettings=$launchSettings; LocalHostIp=$HostAddress; Port=$port;
        FirewallRule=$ruleName; Interface="vEthernet (WSL)"; RemoteAddress="LocalSubnet"} | ConvertTo-Json
    exit 0
}
if ($FirewallOnly) {
    if ($HostAddress -eq "127.0.0.1") { throw "Loopback does not require an inbound WSL firewall rule." }
    # This explicit, administrator-only operation is separate from normal startup.
    if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) { throw "Rule $ruleName already exists; inspect it before changing it." }
    New-NetFirewallRule -Name $ruleName -DisplayName "UAV LLM AirSim from WSL only" `
        -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port `
        -Program $simulatorExe -InterfaceAlias $adapter.InterfaceAlias `
        -RemoteAddress LocalSubnet -Profile Any -EdgeTraversalPolicy Block | Out-Null
    Write-Output "Added ${ruleName}: TCP $port on $($adapter.InterfaceAlias), LocalSubnet only."
    exit 0
}
$existing = Get-Process -Name "Blocks" -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $simulatorExe -or $_.Path -eq $BlocksExe }
if ($existing) { throw "Blocks is already running; close its window before changing the listener." }
New-Item -ItemType Directory -Path $cache -Force | Out-Null
$config | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $launchSettings -Encoding UTF8
Start-Process -FilePath $BlocksExe -WorkingDirectory (Split-Path -Parent $BlocksExe) `
    -ArgumentList @("-ResX=1280", "-ResY=720", "-windowed", "-dx11", "-settings=`"$launchSettings`"")
$deadline = (Get-Date).AddSeconds(60)
do {
    Start-Sleep -Milliseconds 500
    $listener = Get-NetTCPConnection -LocalAddress $HostAddress -LocalPort $port -State Listen -ErrorAction SilentlyContinue
} while (-not $listener -and (Get-Date) -lt $deadline)
if (-not $listener) { throw "Blocks did not listen on ${HostAddress}:$port within 60 seconds." }
Write-Output "Blocks RPC is listening on ${HostAddress}:$port. No flight command was sent."
