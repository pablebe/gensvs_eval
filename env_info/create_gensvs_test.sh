#!/bin/bash

# Exit immediately on error
set -e

# Create the environment
echo "Creating Conda environment 'gensvs_test'..."
conda env create -f "./env_info/gensvs_test.yml"

# Activate the environment
echo "Activating environment..."
# NOTE: This only works in interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gensvs_test

# Set CUDA_HOME environment variable
export CUDA_HOME="${CONDA_PREFIX}"
echo "CUDA_HOME set to: $CUDA_HOME"

# install gensvs package
echo "Installing gensvs package..."
pip install -e "/home/bereuter/experiments/gensvs"