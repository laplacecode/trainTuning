param(
    [string]$RuntimeIdentifier = "win-x64",
    [string]$Configuration = "Release",
    [string]$PythonHome = ""
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$projectPath = Join-Path $repositoryRoot "src\trainTuning.App\trainTuning.App.csproj"
$publishPath = Join-Path $repositoryRoot "artifacts\publish\$RuntimeIdentifier"

dotnet publish $projectPath `
    --configuration $Configuration `
    --runtime $RuntimeIdentifier `
    --self-contained true `
    --output $publishPath `
    -p:PublishSingleFile=false `
    -p:DebugType=None `
    -p:DebugSymbols=false

if ($LASTEXITCODE -ne 0) {
    throw "dotnet publish failed with exit code $LASTEXITCODE"
}

if (-not [string]::IsNullOrWhiteSpace($PythonHome)) {
    $resolvedPythonHome = (Resolve-Path -LiteralPath $PythonHome).Path
    $pythonExecutable = Join-Path $resolvedPythonHome "python.exe"
    if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
        throw "python.exe was not found in PythonHome: $resolvedPythonHome"
    }

    $runtimePath = Join-Path $publishPath "runtime\python"
    New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
    Copy-Item -Path (Join-Path $resolvedPythonHome "*") -Destination $runtimePath -Recurse -Force
}

Write-Host "Publish completed: $publishPath"
if ([string]::IsNullOrWhiteSpace($PythonHome)) {
    Write-Host "Python was not bundled. Pass -PythonHome with a prepared standalone Python/Conda environment."
}
