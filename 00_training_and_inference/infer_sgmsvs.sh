#!/bin/bash

# Bash script to start infer musical mixtures with SGMSVS model.
python ./00_training_and_inference/inference_sgmsvs.py \
    --test_dir "./03_audio_examples/Mixture" \
    --enhanced_dir "./03_audio_examples/SGMSVS" \
    --ckpt "./trained_models/sgmsvs/epoch=510-sdr=7.22.ckpt" \
    --sampler_type "pc" \
    --corrector "ald" \
    --corrector_steps 2 \
    --snr 0.5 \
    --N 45 \
    --device "cuda"  \
    --output_mono \
    --loudness_normalize 
