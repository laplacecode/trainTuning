$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
dotnet run --project (Join-Path $repositoryRoot "src\trainTuning.App\trainTuning.App.csproj")
