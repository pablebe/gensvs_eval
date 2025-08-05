#!/bin/bash

# Bash script to start infer musical mixtures with Mel-RoFo. (S)
python ./00_training_and_inference/inference_baseline.py \
    --test_dir "audio_examples/mixture" \
    --enhanced_dir "audio_examples/separated/MelRoFo-S" \
    --ckpt "trained_models/melroformer_small/melroformer_small_epoch=548-sdr=8.85.ckpt" \
    --device "cuda"  \
    --output_mono \
    --loudness_normalize
