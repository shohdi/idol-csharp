param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location $taskRoot
try {
    $cudaSetting = if ($Cpu) { 'false' } else { 'true' }
    $flavor = if ($Cpu) { 'cpu' } else { 'cuda' }
    dotnet publish src/Idol.Desktop -c Release -r win-x64 --self-contained true "-p:IdolUseCuda=$cudaSetting" -o "artifacts/IdolStudio-$flavor"
    if ($LASTEXITCODE -ne 0) { throw "Publish failed with code $LASTEXITCODE" }
    Write-Host "Open artifacts/IdolStudio-$flavor/Idol.Desktop.exe"
} finally { Pop-Location }
