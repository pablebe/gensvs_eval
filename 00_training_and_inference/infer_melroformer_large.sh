#!/bin/bash

# Bash script to start infer musical mixtures with Mel-RoFo. (L)

python ./00_training_and_inference/inference_melroformer_large.py \
    --config_path "00_training_and_inference/baseline_models/backbones/configs/config_melroformer_large.yaml" \
    --model_path "trained_models/melroformer_large/MelBandRoformer.ckpt" \
    --input_folder "03_audio_examples/Mixture" \
    --store_dir "03_audio_examples/Mel-RoFo-L" \
    --device_ids 0 \
    --output_mono \
    --loudness_normalize 

