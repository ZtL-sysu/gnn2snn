$ErrorActionPreference = 'Stop'
$researchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = (Resolve-Path (Join-Path $researchRoot '..\..')).Path
$condaPython = 'C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe'
Set-Location -LiteralPath $repositoryRoot
foreach ($datasetName in @('MNIST', 'CIFAR10')) {
    foreach ($teacherSeed in @(41, 42, 43)) {
        $validationPath = Join-Path $researchRoot "runs\confirmation\${datasetName}${teacherSeed}"
        $testPath = Join-Path $researchRoot "runs\frozen_test\${datasetName}${teacherSeed}"
        if (-not (Test-Path -LiteralPath (Join-Path $testPath 'DONE.json'))) {
            if (Test-Path -LiteralPath $testPath) { throw "Inspect incomplete test run: $testPath" }
            & $condaPython -X utf8 research/20261114_story/evaluate_frozen_test.py --run $validationPath --output $testPath 2>&1 |
                Tee-Object -FilePath (Join-Path $researchRoot "test_${datasetName}${teacherSeed}.log")
            if ($LASTEXITCODE -ne 0) { throw "Frozen test failed: $datasetName $teacherSeed" }
        }
        $costPath = Join-Path $researchRoot "costs\cost_${datasetName}${teacherSeed}.json"
        if (-not (Test-Path -LiteralPath $costPath)) {
            & $condaPython -X utf8 research/20261114_story/benchmark_cost.py --run $validationPath --output $costPath
            if ($LASTEXITCODE -ne 0) { throw "Cost measurement failed: $datasetName $teacherSeed" }
        }
        $trainPath = Join-Path $researchRoot "training_checks\${datasetName}${teacherSeed}.json"
        if (-not (Test-Path -LiteralPath $trainPath)) {
            New-Item -ItemType Directory -Path (Split-Path -Parent $trainPath) -Force | Out-Null
            & $condaPython -X utf8 research/20261114_story/evaluate_training.py --run $validationPath --output $trainPath
            if ($LASTEXITCODE -ne 0) { throw "Training diagnostic failed: $datasetName $teacherSeed" }
        }
    }
}
