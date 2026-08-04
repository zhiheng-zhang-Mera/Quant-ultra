[CmdletBinding()]
param(
    [string]$VenvPath = "D:\Quant-Ultra\.venv-full",
    [string]$Mirror = "",
    [switch]$SkipTests,
    [switch]$NoPipUpgrade
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Join-Path $RepoRoot "Quant-4"
$SetupTemp = Join-Path $RepoRoot ".tmp\setup"
New-Item -ItemType Directory -Force -Path $SetupTemp | Out-Null
$env:TEMP = $SetupTemp
$env:TMP = $SetupTemp

$ExistingVenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (Test-Path -LiteralPath $ExistingVenvPython) {
    $Launcher = @($ExistingVenvPython)
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $PythonCommand) {
        $Launcher = @($PythonCommand.Source)
    } else {
        $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($null -eq $PyLauncher) { throw "Python 3 was not found in PATH." }
        $Launcher = @($PyLauncher.Source, "-3")
    }
}

$Arguments = @((Join-Path $ProjectRoot "Main\install_deps.py"), "--venv", $VenvPath)
if ($Mirror) { $Arguments += @("--mirror", $Mirror) }
if ($SkipTests) { $Arguments += "--skip-tests" }
if ($NoPipUpgrade) { $Arguments += "--no-pip-upgrade" }

if ($Launcher.Count -gt 1) {
    & $Launcher[0] $Launcher[1..($Launcher.Count - 1)] $Arguments
} else {
    & $Launcher[0] $Arguments
}
if ($LASTEXITCODE -ne 0) { throw "Quant-Ultra setup failed with exit code $LASTEXITCODE." }
Write-Host "Quant-Ultra setup completed. Report: $ProjectRoot\reports\setup\setup_report.json"
