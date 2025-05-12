#!/bin/bash

# Bash script to start training of Mel-RoFo. (S)
# --base_idr: Base directory where moisesdb and musdb18hq are located
# --ckpt: uncomment this to resume training from a checkpoint
python ./00_training_and_inference/train_baseline.py \
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
    --backbone "mel_band_roformer" \
    --batch_size "1" \
    --num_workers "10" \
    --sr "44100" \
    --n_fft "2048" \
    --hop_length "512" \
    --dim "192" \
    --depth "9" \
    --lr "5e-5" \
    --valid_sep_dir "./sgmse-experiments/valid_output_melroformer_fs44100" \
    --audio_log_files "3" "4" "11" "17" "19" "26" "29" "32" \
    --audio_log_interval "5" \
    --wandb_name "mel_band_roformer_fs44100" \
    --run_id "250217_mel_band_roformer_fs44100" \
    --match_input_audio_length \
    --num_eval_files "50" \
    --max_epochs "550" \
    --nolog \
#    --ckpt "./trained_model/mel_band_roformer.ckpt" \
