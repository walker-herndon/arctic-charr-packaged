#!/bin/bash

# Check if Python is installed as "python" or "python3"
if ! { command -v python &> /dev/null || command -v python3 &> /dev/null; }; then
    echo "Python not found. Please install Python."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip not found. Please install pip."
    exit 1
fi

# Check if a path argument is provided
if [ -z "$1" ]; then
    echo "Please provide a path as an argument."
    exit 1
fi

# Create a new Python venv in the specified path
venv_path="$1/.venv"
if command -v python3 &> /dev/null; then
    python_version="python3"
else
    python_version="python"
fi

$python_version -m venv "$venv_path"

cd "$1"

# Activate the venv
source ".venv/bin/activate"

# Install "arctic_charr_matcher" using pip
# pip install arctic_charr_matcher

# Temporarily install from local directory instead of PyPI
pip install /Users/walkerherndon/Documents/CS/CS-Year-5/CS5199/Arctic_charr_re_id/dist/arctic_charr_matcher-0.1.0-py3-none-any.whl

echo "Virtual environment created and activated in $venv_path."
echo "arctic_charr_matcher installed."
