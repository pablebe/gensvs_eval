#!/bin/bash

# Exit immediately on error
set -e

# Create the environment
echo "Creating Conda environment 'gensvs_env'..."
conda env create -f "./env_info/gensvs_env.yml"

# Activate the environment
echo "Activating environment..."
# NOTE: This only works in interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gensvs_env

# Set CUDA_HOME environment variable
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
echo "CUDA_HOME set to: $CUDA_HOME"

# install gensvs package
echo "Installing gensvs package..."
pip install gensvs

# clear cache of old torch extensions, which could cause problems with compiling cpp extensions
echo "Clearing old torch extensions cache..."
conda activate gensvs_env

# Get the current Python version (major.minor)
PYTHON_VERSION=$(python -c "import sys; print(f'py{sys.version_info.major}{sys.version_info.minor}')")

# Define the torch extensions cache directory
TORCH_EXT_DIR="$HOME/.cache/torch_extensions"
echo "Torch extensions directory: $TORCH_EXT_DIR"
# Check if the directory exists
if [ -d "$TORCH_EXT_DIR" ]; then
    echo "Looking for folders matching $PYTHON_VERSION in $TORCH_EXT_DIR"

    # Find and delete matching folders
    MATCHING_DIRS=$(find "$TORCH_EXT_DIR" -maxdepth 1 -type d -name "*$PYTHON_VERSION*")

    if [ -z "$MATCHING_DIRS" ]; then
        echo "No matching directories found."
    else
        echo "Deleting the following directories:"
        echo "$MATCHING_DIRS"
        echo "$MATCHING_DIRS" | xargs rm -rf
        echo "Done."
    fi
else
    echo "Torch extensions directory not found: $TORCH_EXT_DIR"
fi