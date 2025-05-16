#!/bin/bash

# Bash script to compute BSS-Eval, PEASS and non-intrusive metrics for all models.
# the folder ./01_evaluation_and_correlation/evaluation_audio is not part of this repository.
# To obtain the evaluation audio, please run inference for each model.
python ./01_evaluation_and_correlation/gensvs_eval_metrics.py \
                --mixture_dir "./01_evaluation_and_correlation/evaluation_audio/mixture" \
                --target_dir "./01_evaluation_and_correlation/evaluation_audio/target" \
                --separated_dir_sgmsvs_from_scratch "./01_evaluation_and_correlation/evaluation_audio/sgmsvs" \
                --separated_dir_melroform_bigvgan "./01_evaluation_and_correlation/evaluation_audio/melroformer_bigvgan" \
                --separated_dir_melroform_small "./01_evaluation_and_correlation/evaluation_audio/melroformer_small" \
                --separated_dir_melroform_large "./01_evaluation_and_correlation/evaluation_audio/melroformer_large" \
                --separated_dir_htdemucs "./01_evaluation_and_correlation/evaluation_audio/htdemucs" \
                --output_folder "./01_evaluation_and_correlation/evaluation_metrics_short" \
                --sr 44100
