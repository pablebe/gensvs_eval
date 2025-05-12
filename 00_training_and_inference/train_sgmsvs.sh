#!/bin/bash
# Bash script to start training of SGMSVS
# --base_idr: Base directory where moisesdb and musdb18hq are located
# --ckpt: uncomment this to resume training from a checkpoint
python ./00_training_and_inference/train_sgmsvs.py \
        --base_dir /media/DATA/shared/datasets/MSS_datasets \
        --format MSS \
        --samples_per_track 24 \
        --dataset_str musmoisdb \
        --random_mix \
        --duration 5 \
        --add_augmentation \
        --target_str vocals \
        --full_mix_percentage 0.5 \
        --use_musdb_test_as_valid \
        --backbone ncsnpp_48k \
        --audio_log_interval 5 \
        --batch_size 1 \
        --train_mono \
        --num_workers 10 \
        --sr 44100 \
        --num_eval_files 50 \
        --valid_sep_dir ./valid_output_scratch_scale_0.0516_comp_0.334_fs44100 \
        --audio_log_files 3 4 11 17 19 26 29 32 \
        --audio_log_interval 5 \
        --wandb_name sgmsvs_mss_scratch_scale_0.0516_comp_0.334_fs44100 \
        --run_id 250213_1800_sgmvs_scratch_scale_0.0516_comp_0.334_fs44100 \
        --n_fft 2046 \
        --hop_length 512 \
        --spec_factor 0.0516 \
        --spec_abs_exponent 0.334 \
        --sigma-min 0.1 \
        --sigma-max 1.0 \
        --theta 2.0 \
        --max_epoch 550 \
        --nolog \
    #    --ckpt trained_models/sgmsvs/epoch=510-sdr=7.22.ckpt \
