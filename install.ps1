# Check if Python is installed
if (-not (Get-Command python -ErrorAction SilentlyContinue) -and -not (Get-Command python3 -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Please install Python (python or python3)." -ForegroundColor Red
    exit 1
}

# Check if pip is installed
if (-not (Get-Command pip -ErrorAction SilentlyContinue) -and -not (Get-Command pip3 -ErrorAction SilentlyContinue)) {
    Write-Host "pip not found. Please install pip." -ForegroundColor Red
    exit 1
}

# Check if a path argument is provided
if ($args.Count -eq 0) {
    Write-Host "Please provide a path as an argument." -ForegroundColor Red
    exit 1
}

# Create a new Python venv in the specified path
$venvPath = Join-Path -Path $args[0] -ChildPath ".venv"
$pythonExecutable = if (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" } else { "python" }
$pipExecutable = if (Get-Command pip3 -ErrorAction SilentlyContinue) { "pip3" } else { "pip" }
& $pythonExecutable -m venv $venvPath

# Activate the venv
$activateScript = Join-Path -Path $venvPath -ChildPath "Scripts/Activate.ps1"
if (Test-Path $activateScript) {
    . $activateScript
} else {
    Write-Host "Activation script not found. Activate the virtual environment manually." -ForegroundColor Yellow
}

# Install "arctic_charr_matcher" using pip
if (-not (Test-Path $activateScript)) {
    Write-Host "Please activate the virtual environment and run 'pip install arctic_charr_matcher' manually." -ForegroundColor Yellow
} else {
    & $pipExecutable install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ arctic-charr-matcher==0.1.0
    ipython kernel install --user --name=.venv
    Write-Host "arctic_charr_matcher installed." -ForegroundColor Green
}
