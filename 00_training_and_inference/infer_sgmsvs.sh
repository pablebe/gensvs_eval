#!/bin/bash

# Bash script to start infer musical mixtures with SGMSVS model.
python ./00_training_and_inference/inference_sgmsvs.py \
    --test_dir "./audio_examples/mixture" \
    --enhanced_dir "./audio_examples/separated/SGMSVS" \
    --ckpt "./trained_models/sgmsvs/sgmsvs_epoch=510-sdr=7.22.ckpt" \
    --sampler_type "pc" \
    --corrector "ald" \
    --corrector_steps 2 \
    --snr 0.5 \
    --N 45 \
    --device "cuda"  \
    --output_mono \
    --loudness_normalize 
