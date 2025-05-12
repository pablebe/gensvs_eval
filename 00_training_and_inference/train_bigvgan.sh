#!/bin/bash
# Bash script to start training of SGMSVS
# --base_idr: Base directory where moisesdb and musdb18hq are located
# --ckpt: uncomment this to resume training from a checkpoint
python ./00_training_and_inference/train_finetune_bigvgan.py \
        --base_dir "/media/DATA/shared/datasets/MSS_datasets" \
        --format "MSS" \
        --samples_per_track "24" \
        --dataset_str "musmoisdb" \
        --random_mix \
        --add_augmentation \
        --target_str "vocals" \
        --full_mix_percentage "0.5" \
        --use_musdb_test_as_valid \
        --duration "5" \
        --save_audio \
        --validation_interval "10000" \
        --checkpoint_interval "10000" \
        --normalize "not" \
        --valid_sep_dir "./valid_output_melroform_bigvgan_finetuned" \
        --wandb_name "melroform_bigvgan_finetuned" \
        --run_id "melroform_bigvgan_finetuned" \
        --checkpoint_path "./logs/melroform_bigvgan_finetuned" \
        --bigvgan_ckpt_path "./trained_models/bigvgan_finetuned" \
        --bigvgan_config "./00_training_and_inference/bigvgan_utils/configs/bigvgan_v2_44khz_128band_512x.json" \
        --melroformer_ckpt "./trained_models/melroformer_small/epoch=548-sdr=8.85_new.ckpt" \
