param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location $taskRoot
try {
    $tokenFile = Join-Path $taskRoot 'token.txt'
    if ((Test-Path -LiteralPath $tokenFile) -and -not $env:HF_TOKEN_PATH) {
        $env:HF_TOKEN_PATH = $tokenFile
    }
    $cudaSetting = if ($Cpu) { 'false' } else { 'true' }
    dotnet run --project src/Idol.Desktop -c Release "-p:IdolUseCuda=$cudaSetting"
    if ($LASTEXITCODE -ne 0) { throw "Idol exited with code $LASTEXITCODE" }
} finally { Pop-Location }
