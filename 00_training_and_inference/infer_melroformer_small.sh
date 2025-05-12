#!/bin/bash

# Bash script to start infer musical mixtures with Mel-RoFo. (S)
python ./00_training_and_inference/inference_baseline.py \
    --test_dir "03_audio_examples/Mixture" \
    --enhanced_dir "03_audio_examples/Mel-RoFo-S_new" \
    --ckpt "trained_models/melroformer_small/epoch=548-sdr=8.85.ckpt" \
    --device "cuda" \
