$ErrorActionPreference = 'Stop'
$researchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = (Resolve-Path (Join-Path $researchRoot '..\..')).Path
$condaPython = 'C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe'
Set-Location -LiteralPath $repositoryRoot
foreach ($datasetName in @('MNIST', 'CIFAR10')) {
    foreach ($teacherSeed in @(41, 42, 43)) {
        $runPath = Join-Path $researchRoot "runs\confirmation\${datasetName}${teacherSeed}"
        if (Test-Path -LiteralPath (Join-Path $runPath 'DONE.json')) { continue }
        if (Test-Path -LiteralPath $runPath) { throw "Partial output requires inspection: $runPath" }
        & $condaPython -X utf8 research/20261114_story/run_screen.py --dataset $datasetName --seed $teacherSeed --lr 0.001 --val-count 5000 --T 1 2 4 8 --output $runPath 2>&1 |
            Tee-Object -FilePath (Join-Path $researchRoot "confirmation_${datasetName}${teacherSeed}.log")
        if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $datasetName $teacherSeed exit $LASTEXITCODE" }
    }
}
