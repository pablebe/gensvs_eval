#!/bin/bash

# Bash script to start infer musical mixtures with Mel-RoFo. (S)
python ./00_training_and_inference/inference_melroformer_small_bigvgan.py \
    --test_dir "./03_audio_examples/Mixture" \
    --enhanced_dir "./03_audio_examples/Mel-RoFo-S_BigVGAN" \
    --melroformer_ckpt "./trained_models/melroformer_small/epoch=548-sdr=8.85.ckpt" \
    --device "cuda" \
    --bigvgan_config_file "./00_training_and_inference/bigvgan_utils/configs/bigvgan_v2_44khz_128band_512x.json" \
    --bigvgan_checkpoint "./trained_models/bigvgan_finetuned/g_05570000.ckpt" \
    --bigvgan_use_cuda_kernel \
    --output_mono \
    --loudness_normalize
