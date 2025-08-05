#!/bin/bash

# Bash script to start infer musical mixtures with HTDemucs.
python ./00_training_and_inference/inference_baseline.py \
    --test_dir "audio_examples/mixture" \
    --enhanced_dir "audio_examples/separated/HTDemucs" \
    --ckpt "trained_models/htdemucs/epoch=570-sdr=6.38.ckpt" \
    --device "cuda"  \
    --output_mono \
    --loudness_normalize

