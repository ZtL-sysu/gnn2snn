$ErrorActionPreference = 'Stop'
$researchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = (Resolve-Path (Join-Path $researchRoot '..\..')).Path
$condaPython = 'C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe'
Set-Location -LiteralPath $repositoryRoot
$waitStarted = Get-Date
do {
    $ready = $true
    foreach ($datasetName in @('MNIST', 'CIFAR10')) {
        foreach ($teacherSeed in @(41, 42, 43)) {
            foreach ($relativePath in @("runs\frozen_test\${datasetName}${teacherSeed}\DONE.json", "costs\cost_${datasetName}${teacherSeed}.json", "training_checks\${datasetName}${teacherSeed}.json")) {
                if (-not (Test-Path -LiteralPath (Join-Path $researchRoot $relativePath))) { $ready = $false }
            }
        }
    }
    if (-not $ready) {
        if (((Get-Date) - $waitStarted).TotalMinutes -gt 60) { throw 'Frozen endpoint queue has not completed; inspect it before starting diagnostics' }
        Start-Sleep -Seconds 10
    }
} until ($ready)
foreach ($datasetName in @('MNIST', 'CIFAR10')) {
    foreach ($teacherSeed in @(41, 42, 43)) {
        $validationPath = Join-Path $researchRoot "runs\confirmation\${datasetName}${teacherSeed}"
        $diagnosticPath = Join-Path $researchRoot "runs\ideal\${datasetName}${teacherSeed}"
        if (Test-Path -LiteralPath (Join-Path $diagnosticPath 'DONE.json')) { continue }
        if (Test-Path -LiteralPath $diagnosticPath) { throw "Inspect incomplete diagnostic: $diagnosticPath" }
        & $condaPython -X utf8 research/20261114_story/ideal_site_diagnostic.py --run $validationPath --output $diagnosticPath 2>&1 |
            Tee-Object -FilePath (Join-Path $researchRoot "ideal_${datasetName}${teacherSeed}.log")
        if ($LASTEXITCODE -ne 0) { throw "Ideal-site diagnostic failed: $datasetName $teacherSeed" }
    }
}
