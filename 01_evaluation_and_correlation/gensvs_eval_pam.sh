#!/bin/bash

# Bash script to compute PAM metric of all models.
# the folder ./01_evaluation_and_correlation/evaluation_audio is not part of this repository.
# To obtain the evaluation audio, please run inference for each model.
python ./01_evaluation_and_correlation/gensvs_eval_pam.py \
                --mixture_dir "./gensvs_eval_data/gensvs_eval_audio_and_embeddings/mixture" \
                --separated_dir_sgmsvs_from_scratch "./gensvs_eval_data/gensvs_eval_audio_and_embeddings/sgmsvs" \
                --separated_dir_melroform_bigvgan "./gensvs_eval_data/gensvs_eval_audio_and_embeddings/melroformer_bigvgan" \
                --separated_dir_melroform_small "./gensvs_eval_data/gensvs_eval_audio_and_embeddings/melroformer_small" \
                --separated_dir_melroform_large "./gensvs_eval_data/gensvs_eval_audio_and_embeddings/melroformer_large" \
                --separated_dir_htdemucs "./gensvs_eval_data/gensvs_eval_audio_and_embeddings/htdemucs" \
                --output_folder "./01_evaluation_and_correlation/evaluation_metrics" \