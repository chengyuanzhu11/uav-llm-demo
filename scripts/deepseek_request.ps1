$ErrorActionPreference = "Stop"
try {
    [Console]::InputEncoding = [System.Text.Encoding]::UTF8
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $request = [Console]::In.ReadToEnd() | ConvertFrom-Json
    $response = Invoke-WebRequest -Uri ([string]$request.url) -Method Post `
        -Headers @{ Authorization = "Bearer $($request.api_key)" } `
        -Body ([string]$request.body) -ContentType "application/json; charset=utf-8" `
        -TimeoutSec ([int]$request.timeout) -UseBasicParsing
    @{ status = [int]$response.StatusCode; body = [string]$response.Content } |
        ConvertTo-Json -Compress -Depth 4
} catch {
    if ($_.Exception.Response) {
        @{ status = [int]$_.Exception.Response.StatusCode } | ConvertTo-Json -Compress
    } elseif ($_.Exception.GetBaseException() -is [System.Threading.Tasks.TaskCanceledException] -or
              ($_.Exception -is [System.Net.WebException] -and
               $_.Exception.Status -eq [System.Net.WebExceptionStatus]::Timeout)) {
        '{"error":"timeout"}'
    } else {
        '{"error":"connection"}'
    }
}
