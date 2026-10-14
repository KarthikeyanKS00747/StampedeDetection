param(
    [int]$ImageSize = 1024,
    [switch]$Tiled
)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runOutput = Join-Path $projectDirectory ('outputs/training/gpu_person_' + $runStamp)
Push-Location -LiteralPath $projectDirectory
try {
    $runArguments = @('-m', 'training.gpu_person_baseline', '--output-dir', $runOutput, '--device', '0', '--imgsz', [string]$ImageSize)
    if ($Tiled) { $runArguments += '--tiled' }
    & python @runArguments
    if ($LASTEXITCODE -ne 0) { throw 'GPU experiment failed. Check the error above; saved checkpoints may be available.' }
    Write-Host ('Completed experimental run: ' + $runOutput)
    Write-Host 'Research baseline only. This did not fine-tune YOLO or validate stampede risk.'
} finally {
    Pop-Location
}
