#!/bin/bash

# Bash script to start trianing of HTDemucs
# --base_idr: Base directory where moisesdb and musdb18hq are located
# --ckpt: uncomment this to resume training from a checkpoint
# --nolog: comment this to enable logging
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
    --backbone "htdemucs" \
    --loss_type "l1_loss" \
    --lr "1e-4" \
    --bypass_lr_scheduler "true" \
    --batch_size "6" \
    --num_workers "14" \
    --sr "48000" \
    --masked_mse_coarse "true" \
    --masked_mse_q "0.95" \
    --channels "48" \
    --growth "2" \
    --num_subbands "1" \
    --nfft "4096" \
    --wiener_iters "0" \
    --end_iters "0" \
    --wiener_residual "false" \
    --cac "true" \
    --depth "4" \
    --rewrite "true" \
    --multi_freqs_depth "3" \
    --freq_emb "0.2" \
    --emb_scale "10" \
    --emb_smooth "true" \
    --kernel_size "8" \
    --stride "4" \
    --time_stride "2" \
    --context "1" \
    --context_enc "0" \
    --norm_starts "4" \
    --norm_groups "4" \
    --dconv_mode "3" \
    --dconv_depth "2" \
    --dconv_comp "8" \
    --dconv_init "1e-3" \
    --bottom_channels "768" \
    --t_layers "5" \
    --t_hidden_scale "4.0" \
    --t_heads "8" \
    --t_dropout "0.0" \
    --t_layer_scale "true" \
    --t_gelu "true" \
    --t_emb "sin" \
    --t_max_positions "10000" \
    --t_max_period "10000.0" \
    --t_weight_pos_embed "1.0" \
    --t_cape_mean_normalize "true" \
    --t_cape_augment "true" \
    --t_cape_glob_loc_scale "5000.0" "1.0" "1.4" \
    --t_sin_random_shift "0" \
    --t_norm_in "true" \
    --t_norm_in_group "false" \
    --t_group_norm "false" \
    --t_norm_first "true" \
    --t_norm_out "true" \
    --t_weight_decay "0.0" \
    --t_sparse_self_attn "false" \
    --t_sparse_cross_attn "false" \
    --t_mask_type "diag" \
    --t_mask_random_seed "42" \
    --t_sparse_attn_window "400" \
    --t_global_window "100" \
    --t_sparsity "0.95" \
    --t_auto_sparsity "false" \
    --t_cross_first "false" \
    --rescale "0.1" \
    --valid_sep_dir "./valid_output_htdemucs_large_l1_loss" \
    --audio_log_files "3" "4" "11" "17" "19" "26" "29" "32" \
    --audio_log_interval "5" \
    --wandb_name "htdemucs_large_l1_loss" \
    --run_id "2502041728_htdemucs_large_l1_loss" \
    --num_eval_files "50" \
    --nolog \
#    --ckpt "./logs/2502041728_htdemucs_large_l1_loss/last.ckpt"