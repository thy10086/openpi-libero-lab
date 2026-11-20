param(
    [string]$RunId = (Get-Date -Format 'yyyyMMdd-HHmmss'),
    [ValidateRange(1, 50)][int]$Trials = 3,
    [ValidateRange(1, 1000)][int]$Epochs = 25,
    [switch]$Build
)

$ErrorActionPreference = 'Stop'
if ($RunId -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$') {
    throw 'RunId must contain only letters, numbers, underscores or hyphens.'
}

$project = $PSScriptRoot
$image = 'openpi-trajectory-lab:cpu'
$run = "lab_runs/$RunId"
$sample = 'lab_data/libero_spatial_0.parquet'
$revision = 'a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4'
if (Test-Path -LiteralPath (Join-Path $project $run)) { throw "Run already exists: $run" }
if (!(Test-Path -LiteralPath (Join-Path $project $sample))) { throw 'LIBERO demonstration sample is missing.' }
if (!(Test-Path -LiteralPath (Join-Path $project 'third_party/libero/libero/libero/init_files'))) {
    throw 'Initialize the LIBERO submodule before running the demo.'
}

$existing = & docker image ls -q $image
if ($LASTEXITCODE -ne 0) { throw 'Docker engine unavailable.' }
if ($Build -or !$existing) {
    & docker build -f (Join-Path $project 'trajectory_lab/Dockerfile') -t $image (Join-Path $project 'trajectory_lab')
    if ($LASTEXITCODE -ne 0) { throw 'CPU demo image failed to build.' }
}

$mount = "type=bind,source=$project,target=/app"
function Invoke-Lab {
    param([string[]]$Arguments)
    & docker run --rm --network none --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --mount $mount $image @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Lab command failed: $($Arguments -join ' ')" }
}

Invoke-Lab @('python', '-m', 'unittest', 'trajectory_lab.test_lab', 'trajectory_lab.test_protocol', '-v')
Invoke-Lab @('python', '-m', 'trajectory_lab', 'audit', '--input', $sample, '--fps', '10',
    '--dataset', 'lerobot/libero', '--revision', $revision, '--strict', '--output', "$run/audit.json")
Invoke-Lab @('python', '-m', 'trajectory_lab', 'train-bc', '--input', $sample, '--fps', '10',
    '--dataset', 'lerobot/libero', '--revision', $revision, '--task-index', '34',
    '--epochs', "$Epochs", '--output', "$run/bc")
Invoke-Lab @('python', '-m', 'trajectory_lab', 'toy', '--trials', '5', '--execute-steps', '5', '1', '10',
    '--output', "$run/toy")
Invoke-Lab @('python', '-m', 'trajectory_lab', 'libero', '--policy', 'zero', '--trials', "$Trials",
    '--execute-steps', '5', '--record-video', '--output', "$run/zero")
Invoke-Lab @('python', '-m', 'trajectory_lab', 'libero', '--policy', 'bc', '--trials', "$Trials",
    '--checkpoint', "$run/bc/policy.pt", '--execute-steps', '5', '10', '--record-video',
    '--output', "$run/closed_loop")
Invoke-Lab @('python', '-m', 'trajectory_lab.render_demo', $run)
Write-Output (Join-Path $project "$run/index.html")
