# Copyright (c) 2025
#   Licensed under the MIT license.

# Adapted from https://github.com/NVIDIA/BigVGAN/tree/main under the MIT license.
#   LICENSE is in incl_licenses directory.


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import itertools
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set your CUDA device here
os.environ["WANDB__SERVICE_WAIT"] = "300"

import time
import argparse
import copy
import json
import torch
import torch.nn.functional as F
import numpy as np
import wandb
import tqdm
import auraloss
import torch.multiprocessing as mp
import torchaudio as ta

from baseline_models.MSS_mask_model import MaskingModel #.MSS_model import ScoreModel

from torch import einsum
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from bigvgan_utils.env import AttrDict, build_env
from bigvgan_utils.meldataset import spectral_normalize_torch, MelDataset, mel_spectrogram, get_dataset_filelist, MAX_WAV_VALUE
from sgmsvs.data_module import MSSSpecs
from bigvgan_utils.bigvgan import BigVGAN
from bigvgan_utils.discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiBandDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from bigvgan_utils.loss import (
    feature_loss,
    generator_loss,
    discriminator_loss,
    MultiScaleMelSpectrogramLoss,
)
from bigvgan_utils.utils import (
    plot_spectrogram,
    plot_spectrogram_clipped,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_audio,
)
from pesq import pesq
from tqdm import tqdm
#from torch.serialization import add_safe_globals
#from sgmsvs.data_module import SpecsDataModule  # new path

#add_safe_globals([SpecsDataModule])
#torch.serialization.safe_globals([SpecsDataModule])

wandb.login()
torch.backends.cudnn.benchmark = False

def get_argparse_groups(parser,args):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups

def train(rank, a, h):
    if h.num_gpus > 1:
        # initialize distributed
        init_process_group(
            backend=h.dist_config["dist_backend"],
            init_method=h.dist_config["dist_url"],
            world_size=h.dist_config["world_size"] * h.num_gpus,
            rank=rank,
        )

    # Set seed and device
    torch.cuda.manual_seed(h.seed)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank:d}")


    # Define BigVGAN generator
    generator = BigVGAN(h).to(device)

    # Define discriminators. MPD is used by default
    mpd = MultiPeriodDiscriminator(h).to(device)

    # Define additional discriminators. BigVGAN-v1 uses UnivNet's MRD as default
    # New in BigVGAN-v2: option to switch to new discriminators: MultiBandDiscriminator / MultiScaleSubbandCQTDiscriminator
    if h.get("use_mbd_instead_of_mrd", False):  # Switch to MBD
        print(
            "[INFO] using MultiBandDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
        )
        # Variable name is kept as "mrd" for backward compatibility & minimal code change
        mrd = MultiBandDiscriminator(h).to(device)
    elif h.get("use_cqtd_instead_of_mrd", False):  # Switch to CQTD
        print(
            "[INFO] using MultiScaleSubbandCQTDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
        )
        mrd = MultiScaleSubbandCQTDiscriminator(h).to(device)
    else:  # Fallback to original MRD in BigVGAN-v1
        mrd = MultiResolutionDiscriminator(h).to(device)

    # New in BigVGAN-v2: option to switch to multi-scale L1 mel loss
    if h.get("use_multiscale_melloss", False):
        print(
            "[INFO] using multi-scale Mel l1 loss of BigVGAN-v2 instead of the original single-scale loss"
        )
        fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
            sampling_rate=h.sampling_rate
        )  # NOTE: accepts waveform as input
    else:
        fn_mel_loss_singlescale = F.l1_loss

    # Print the model & number of parameters, and create or scan the latest checkpoint from checkpoints directory
    if rank == 0:
        print(generator)
        print(mpd)
        print(mrd)
        print(f"Generator params: {sum(p.numel() for p in generator.parameters())}")
        print(f"Discriminator mpd params: {sum(p.numel() for p in mpd.parameters())}")
        print(f"Discriminator mrd params: {sum(p.numel() for p in mrd.parameters())}")
        os.makedirs(a.bigvgan_ckpt_path, exist_ok=True)
        print(f"Checkpoints directory: {a.bigvgan_ckpt_path}")

    if os.path.isdir(a.bigvgan_ckpt_path):
        # New in v2.1: If the step prefix pattern-based checkpoints are not found, also check for renamed files in Hugging Face Hub to resume training
        cp_g = scan_checkpoint(
            a.bigvgan_ckpt_path, prefix="g_", renamed_file="bigvgan_generator.pt"
        )
        cp_do = scan_checkpoint(
            a.bigvgan_ckpt_path,
            prefix="do_",
            renamed_file="bigvgan_discriminator_optimizer.pt",
        )

    # Load the latest checkpoint if exists
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        mrd.load_state_dict(state_dict_do["mrd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]
    
    # load melroformer and bigvgan from checkpoint    
    melroform_bigvgan_model = MaskingModel.load_from_checkpoint(a.melroformer_ckpt, map_location='cpu')
    # clone melrformer model
    melroformer = copy.deepcopy(melroform_bigvgan_model.dnn)
    melroformer.eval()
    melroformer = melroformer.to("cuda")
    if a.train_mono:
        melroformer.stereo = False
    del melroform_bigvgan_model
    #freeze melroformer model!
    for param in melroformer.parameters():
        param.requires_grad = False

    # Initialize DDP, optimizers, and schedulers
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    # 80% of 500000 steps = 400000 to go from 1.35e-5 to 1e-5 original decay: 0.9999996,
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    if a.window == "sqrthann":
        win = torch.sqrt(torch.hann_window(h.win_size))
    elif a.window == "hann":
        win = torch.hann_window(h.win_size)
    else:
        win = torch.ones(h.win_size)
        
    stft_kwargs = {'n_fft':h.n_fft, 'hop_length':h.hop_size, 'center':True, 'window':win, 'win_length':h.win_size, 'return_complex': True}
    trainset = MSSSpecs(data_dir=a.base_dir, 
                        samples_per_track=a.samples_per_track, 
                        subset='train',
                        dataset_str=a.dataset_str, 
                        valid_split = a.valid_split, 
                        rand_seed=a.rand_seed,
                        target_str=a.target_str, 
                        random_mix_flag=a.random_mix, 
                        augmentation_flag=a.add_augmentation, 
                        enforce_full_mix_percentage=a.full_mix_percentage, 
                        duration=a.duration, 
                        dummy=a.dummy, 
                        normalize=a.normalize, 
                        train_mono=a.train_mono, 
                        num_frames= int(np.ceil(h.sampling_rate * a.duration / h.hop_size)),
                        stft_kwargs=stft_kwargs
                        )

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )
    ## get complexity info
    
    from ptflops import get_model_complexity_info

    macs_melroformer, _ = get_model_complexity_info(
          melroformer,
          (2,1*h.sampling_rate),  # Input size without batch dim
          as_strings=False,
          print_per_layer_stat=True,
          verbose=True
    )
    
    mel_in = mel_spectrogram(
                                torch.rand((2, 1*h.sampling_rate)),
                                h.n_fft,
                                h.num_mels,
                                h.sampling_rate,
                                h.hop_size,
                                h.win_size,
                                h.fmin,
                                h.fmax_for_loss,
                            )
    
    mel_in = mel_in.squeeze().to(device)
    
    macs_bigvgan, params_bigvgan = get_model_complexity_info(
          generator,
          (mel_in.shape[1], mel_in.shape[2]),  # Input size without batch dim
          as_strings=False,
          print_per_layer_stat=True,
          verbose=True
    )
     
    print("PTFLOPS model summary for: melroformer plus bigvgan ")
    print(f"MACs/s: {macs_melroformer / 1e9:.2f} + {macs_bigvgan / 1e9:.2f}B")
    print(f"Parameters: {params_bigvgan / 1e6:.2f} M")

    if rank == 0:
        validset = MSSSpecs(data_dir=a.base_dir, 
                            samples_per_track=a.samples_per_track, 
                            subset='test',
                            dataset_str=a.dataset_str, 
                            valid_split = 0.0, 
                            rand_seed=1234,
                            target_str=a.target_str, 
                            random_mix_flag=False, 
                            augmentation_flag=False, 
                            enforce_full_mix_percentage=0, 
                            duration=a.duration, 
                            dummy=a.dummy,      
                            normalize='not', 
                            train_mono=a.train_mono,
                            num_frames= int(np.ceil(h.sampling_rate * a.duration / h.hop_size)),
                            stft_kwargs=stft_kwargs
)


        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
        )

        # initialize wandb logging
        wandb.init(project="sgmse-MSS", name=a.wandb_name, dir="logs", id=a.run_id, resume="allow")
        if a.save_audio:  # Also save audio to disk if --save_audio is set to True
            os.makedirs(os.path.join(a.valid_sep_dir), exist_ok=True)

    """
    Validation loop, "mode" parameter is automatically defined as (seen or unseen)_(name of the dataset).
    If the name of the dataset contains "nonspeech", it skips PESQ calculation to prevent errors 
    """

    def validate(rank, a, h, loader, mode="seen"):
        assert rank == 0, "validate should only run on rank=0"
        generator.eval()
        torch.cuda.empty_cache()

        val_err_tot = 0
        val_pesq_tot = 0
        val_mrstft_tot = 0

        # Modules for evaluation metrics
        pesq_resampler = ta.transforms.Resample(h.sampling_rate, 16000).cuda()
        loss_mrstft = auraloss.freq. MultiResolutionSTFTLoss(
                                                    fft_sizes=[1024, 2048, 8192],
                                                    hop_sizes=[256, 512, 2048],
                                                    win_lengths=[1024, 2048, 8192],
                                                    scale="mel",
                                                    n_bins=128,
                                                    sample_rate=h.sampling_rate,
                                                    perceptual_weighting=True,
                                                )

        if a.save_audio:  # Also save audio to disk if --save_audio is set to True
            os.makedirs(
                os.path.join(a.valid_sep_dir, f"target"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(a.valid_sep_dir, f"mixture"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(a.valid_sep_dir, f"separated_melroformer_and_bigvgan"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(a.valid_sep_dir, f"separated_melroformer"),
                exist_ok=True,
            )

        with torch.no_grad():
            print(f"step {steps} {mode} speaker validation...")

            # Loop over validation set and compute metrics
            for j, batch in enumerate(tqdm(loader)):
                stft_target, _, audio_y, audio_x = batch # here y is target and x is mixture
              

                y = audio_y.to(device)
                x = audio_x.to(device)
                with torch.no_grad():
                    norm_fac = x.abs().max()
                    sep_audio = melroformer(x/norm_fac)
                    sep_audio = sep_audio * norm_fac
                    mel_sep = mel_spectrogram(
                                                sep_audio.squeeze(),
                                                h.n_fft,
                                                h.num_mels,
                                                h.sampling_rate,
                                                h.hop_size,
                                                h.win_size,
                                                h.fmin,
                                                h.fmax_for_loss,
                                            )
                    
                    mel_sep = mel_sep.squeeze().to(device)
                

                    y_mel = mel_spectrogram(
                                            audio_y.squeeze(),
                                            h.n_fft,
                                            h.num_mels,
                                            h.sampling_rate,
                                            h.hop_size,
                                            h.win_size,
                                            h.fmin,
                                            h.fmax_for_loss,
                                            )

                    if hasattr(generator, "module"):
                        y_g_hat = generator.module(mel_sep)
                    else:
                        y_g_hat = generator(mel_sep)
                    y_mel = y_mel.to(device, non_blocking=True)
                    y_g_hat_mel = mel_spectrogram(
                        y_g_hat.squeeze(1),
                        h.n_fft,
                        h.num_mels,
                        h.sampling_rate,
                        h.hop_size,
                        h.win_size,
                        h.fmin,
                        h.fmax_for_loss,
                    )
                    min_t = min(y_mel.size(-1), y_g_hat_mel.size(-1))
                    val_err_tot += F.l1_loss(y_mel[...,:min_t].squeeze(), y_g_hat_mel[...,:min_t].squeeze()).item()

                    if (
                        not "nonspeech" in mode and not(mode=='seen_musmoisdb')     
                    ):  # Skips if the name of dataset (in mode string) contains "nonspeech"
                        # Resample to 16000 for pesq
                        y_16k = pesq_resampler(y)
                        y_g_hat_16k = pesq_resampler(y_g_hat.squeeze(1))
                        y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
                        y_g_hat_int_16k = (
                            (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
                        )
                        val_pesq_tot += pesq(16000, y_int_16k, y_g_hat_int_16k, "wb")

                    # MRSTFT calculation
                    min_t = min(y.size(-1), y_g_hat.size(-1))
                    val_mrstft_tot += loss_mrstft(y_g_hat[...,:min_t], y[...,:min_t]).item()

                    # Log audio and figures to Tensorboard
                    if j % a.eval_subsample == 0:  # Subsample every nth from validation set
    
                        if (
                            a.save_audio
                        ):  # Also save audio to disk if --save_audio is set to True
                            save_audio(
                                y_g_hat.squeeze().cpu().T,
                                os.path.join(
                                    a.valid_sep_dir,
                                    f"separated_melroformer_and_bigvgan",
                                    f"separated_fileid_{j}.wav",
                                ),
                                h.sampling_rate,
                            )
                            save_audio(
                                sep_audio.squeeze().cpu().T,
                                os.path.join(
                                    a.valid_sep_dir,
                                    f"separated_melroformer",
                                    f"separated_fileid_{j}.wav",
                                ),
                                h.sampling_rate,
                            )
                            save_audio(
                                audio_y.squeeze().cpu().T,
                                os.path.join(
                                    a.valid_sep_dir,
                                    f"target",
                                    f"target_fileid_{j}.wav",
                                ),
                                h.sampling_rate,
                            )
                            save_audio(
                                audio_x.squeeze().cpu().T,
                                os.path.join(
                                    a.valid_sep_dir,
                                    f"mixture",
                                    f"mixture_fileid_{j}.wav",
                                ),
                                h.sampling_rate,
                            )

                        """
                        Visualization of spectrogram difference between GT and synthesized audio, difference higher than 1 is clipped for better visualization.
                        """

            val_err = val_err_tot / (j + 1)
            val_mrstft = val_mrstft_tot / (j + 1)
                # Log evaluation metrics to Tensorboard
            wandb.log({f"validation_{mode}/mel_spec_error":val_err, f"validation_{mode}/mrstft":val_mrstft, 'step':steps}, step=steps)

        generator.train()

    # If the checkpoint is loaded, start with validation loop
    if steps != 0 and rank == 0 and not a.debug:
        if not a.skip_seen:
            validate(
                rank,
                a,
                h,
                validation_loader,
                mode=f"seen_{train_loader.dataset.dataset_str}",
            )
    # Exit the script if --evaluate is set to True
    if a.evaluate:
        exit()

    # Main training loop
    generator.train()
    mpd.train()
    mrd.train()
    torch.cuda.empty_cache()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print(f"Epoch: {epoch + 1}")

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in  tqdm(enumerate(train_loader), total=len(train_loader), desc='Finetunig BigVGAN, epoch: '+str(epoch)):
        
            if rank == 0:
                start_b = time.time()
            
            stft_target, _, audio_y, audio_x = batch

            y = audio_y.to(device)
            x = audio_x.to(device)
            
            #TODO: do segmentation of audio here and train on smaller segments! => everything gan under a for loop.
            with torch.no_grad():
                norm_fac = x.abs().max()
                sep_audio = melroformer(x/norm_fac)
                sep_audio = sep_audio * norm_fac
                
            sep_audio_seg = sep_audio.squeeze().unfold(-1, h.segment_size, h.segment_size).permute(1,0,2)
            sep_audio_seg = sep_audio_seg.reshape(sep_audio_seg.size(0)*sep_audio_seg.size(1), h.segment_size).unsqueeze(1)
            y_seg = y.squeeze().unfold(-1, h.segment_size, h.segment_size).permute(1,0,2)
            y_seg = y_seg.reshape(y_seg.size(0)*y_seg.size(1), h.segment_size).unsqueeze(1)
            seg_ct = 0

            for sep_seg, targ_seg in zip(sep_audio_seg, y_seg):

                seg_ct += 1
                mel_sep = mel_spectrogram(
                            sep_seg,
                            h.n_fft,
                            h.num_mels,
                            h.sampling_rate,
                            h.hop_size,
                            h.win_size,
                            h.fmin,
                            h.fmax_for_loss,
                        )   
                    
                mel_sep = mel_sep.to(device)
                
                targ_seg_mel = mel_spectrogram(
                            targ_seg,
                            h.n_fft,
                            h.num_mels,
                            h.sampling_rate,
                            h.hop_size,
                            h.win_size,
                            h.fmin,
                            h.fmax_for_loss,
                            )
                            
                y_g_hat = generator(mel_sep)
                        
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1),
                    h.n_fft,
                    h.num_mels,
                    h.sampling_rate,
                    h.hop_size,
                    h.win_size,
                    h.fmin,
                    h.fmax_for_loss,
                )

                optim_d.zero_grad()

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(targ_seg.unsqueeze(0), y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )

                # MRD
                y_ds_hat_r, y_ds_hat_g, _, _ = mrd(targ_seg.unsqueeze(0), y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )

                loss_disc_all = loss_disc_s + loss_disc_f

                # Set clip_grad_norm value
                clip_grad_norm = h.get("clip_grad_norm", 1000.0)  # Default to 1000

                # Whether to freeze D for initial training steps
                if steps >= a.freeze_step:
                    loss_disc_all.backward()
                    grad_norm_mpd = torch.nn.utils.clip_grad_norm_(
                        mpd.parameters(), clip_grad_norm
                    )
                    grad_norm_mrd = torch.nn.utils.clip_grad_norm_(
                        mrd.parameters(), clip_grad_norm
                    )
                    optim_d.step()
                else:
                    print(
                        f"[WARNING] skipping D training for the first {a.freeze_step} steps"
                    )
                    grad_norm_mpd = 0.0
                    grad_norm_mrd = 0.0

                # Generator
                optim_g.zero_grad()

                # L1 Mel-Spectrogram Loss
                lambda_melloss = h.get(
                    "lambda_melloss", 45.0
                )  # Defaults to 45 in BigVGAN-v1 if not set
                if h.get("use_multiscale_melloss", False):  # uses wav <y, y_g_hat> for loss
                    loss_mel = fn_mel_loss_multiscale(targ_seg.unsqueeze(0), y_g_hat) * lambda_melloss
                else:  # Uses mel <y_mel, y_g_hat_mel> for loss
                    loss_mel = fn_mel_loss_singlescale(targ_seg_mel, y_g_hat_mel) * lambda_melloss

                # MPD loss
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(targ_seg.unsqueeze(0), y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

                # MRD loss
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(targ_seg.unsqueeze(0), y_g_hat)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

                if steps >= a.freeze_step:
                    loss_gen_all = (
                        loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                    )
                else:
                    print(
                        f"[WARNING] using regression loss only for G for the first {a.freeze_step} steps"
                    )
                    loss_gen_all = loss_mel

                loss_gen_all.backward()
                grad_norm_g = torch.nn.utils.clip_grad_norm_(
                    generator.parameters(), clip_grad_norm
                )
                optim_g.step()
                

                if rank == 0:
                    # STDOUT logging
                    if steps % a.stdout_interval == 0:

                        mel_error = (
                            loss_mel.item() / lambda_melloss
                        )  # Log training mel regression loss to stdout
                        print(
                            f"Steps: {steps:d}, "
                            f"Gen Loss Total: {loss_gen_all.item():4.3f}, "
                            f"Mel Error: {mel_error:4.3f}, "
                            f"s/b: {time.time() - start_b:4.3f} "
                            f"lr: {optim_g.param_groups[0]['lr']:4.7f} "
                            f"grad_norm_g: {grad_norm_g:4.3f}"
                        )

                    # Checkpointing
                    if steps % a.checkpoint_interval == 0 and steps != 0:
                        checkpoint_path = f"{a.checkpoint_path}/g_{steps:08d}"
                        save_checkpoint(
                            checkpoint_path,
                            {
                                "generator": (
                                    generator.module if h.num_gpus > 1 else generator
                                ).state_dict()
                            },
                        )
                        checkpoint_path = f"{a.checkpoint_path}/do_{steps:08d}"
                        save_checkpoint(
                            checkpoint_path,
                            {
                                "mpd": (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                "mrd": (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
                                "optim_g": optim_g.state_dict(),
                                "optim_d": optim_d.state_dict(),
                                "steps": steps,
                                "epoch": epoch,
                            },
                        )

                    # Tensorboard summary logging
                    if steps % a.summary_interval == 0:
                        mel_error = (
                            loss_mel.item() / lambda_melloss
                        )  # Log training mel regression loss to tensorboard
                        wandb.log({"gen_loss_total":loss_gen_all.item(),
                                    "mel_spec_error":mel_error,
                                    "fm_loss_mpd":loss_fm_f.item(),
                                    "gen_loss_mpd":loss_gen_f.item(),
                                    "disc_loss_mpd":loss_disc_f.item(),
                                    "grad_norm_mpd":grad_norm_mpd,
                                    "fm_loss_mrd":loss_fm_s.item(),
                                    "gen_loss_mrd":loss_gen_s.item(),
                                    "disc_loss_mrd":loss_disc_s.item(),
                                    "grad_norm_mrd":grad_norm_mrd,
                                    "grad_norm_g":grad_norm_g,
                                    "learning_rate_d":scheduler_d.get_last_lr()[0],
                                    "learning_rate_g":scheduler_g.get_last_lr()[0],
                                    "epoch":epoch + 1,
                                    },step=steps)


                    # Validation
                    if steps % a.validation_interval == 0:


                        # Seen and unseen speakers validation loops
                        if not a.debug and steps != 0:
                            validate(
                                rank,
                                a,
                                h,
                                validation_loader,
                                mode=f"seen_{train_loader.dataset.dataset_str}",
                            )

                steps += 1

                # BigVGAN-v2 learning rate scheduler is changed from epoch-level to step-level
                scheduler_g.step()
                scheduler_d.step()

        #save last checkpoint after each epoch
        checkpoint_path = f"{a.checkpoint_path}/g_last_ckpt"
        save_checkpoint(
            checkpoint_path,
            {
                "generator": (
                    generator.module if h.num_gpus > 1 else generator
                ).state_dict()
            },
        )
        checkpoint_path = f"{a.checkpoint_path}/do_last_ckpt"
        save_checkpoint(
            checkpoint_path,
            {
                "mpd": (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                "mrd": (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "steps": steps,
                "epoch": epoch,
            },
        )
        if rank == 0:
            print(
                f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n"
            )


def main():
    print("Initializing Training Process..")

    parser = argparse.ArgumentParser()
    
    # svs data parsing args
    parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
    parser.add_argument("--format", type=str, choices=("MSS", "default", "reverb"), default="MSS", help="Read file paths according to file naming format.")
    parser.add_argument("--dataset_str", type=str, choices=("musdb", "moisesdb", "musmoisdb"), default="musdb", help="Dataset to use. Default is musdb.")
    parser.add_argument("--target_str", type=str, choices=("vocals", "bass", "drums", "other", "all"), default="vocals", help="Target source to extract. 'vocals' by default.")
    parser.add_argument("--samples_per_track", type=int, default=1, help="How often each track is used in the dataset. Allows for oversampling of the dataset. 1 by default")
    parser.add_argument("--valid_split", type=float, default=0.1, help="Fraction of the training set to use for validation. 0.1 by default.")
    parser.add_argument("--use_musdb_test_as_valid", action="store_true", default=False, help="Use musdb test set as validation")
    parser.add_argument("--random_mix", action="store_true", default=False, help="Set for random mixing of sources.")
    parser.add_argument("--add_augmentation", action="store_true", default=False, help="Set for data augmentation.")
    parser.add_argument("--full_mix_percentage", type=float, default=0.7, help="Percentage of data with non-silent sources. 0.7 by default.")
    parser.add_argument("--random_seed", default=13, type=int, help="Random seed that defines train/valid split.")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of the audio files. 5 seconds by default.")
    parser.add_argument("--rand_seed", type=int, default=13, help="Random seed for dataset creation. 13 by default.")
    parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
    parser.add_argument("--train_mono", action="store_true", help="Use only the first channel of the audio files.")
    parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
    parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
    parser.add_argument("--wandb_name", type=str, default=None, help="Name for wandb logger. If not set, a random name is generated.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
    parser.add_argument("--run_id", type=str, default=None, help="Set run id so distributed training is logged on same run")
    parser.add_argument("--nolog", action='store_true', help="Turn off logging.")
    parser.add_argument("--valid_sep_dir", type=str, default=None, help="The directory in which separated validation examples are stored.")


    parser.add_argument("--bigvgan_ckpt_path",type=str, default="bigvgan_ckpt_path", help="Path to the BigVGAN checkpoint.")
    parser.add_argument("--bigvgan_config", type=str, default="bigvgan_config", help="Path to the BigVGAN config file.")
    parser.add_argument("--melroformer_ckpt", type=str, default="melroformer_ckpt", help="Path to the Melroformer checkpoint.")
    parser.add_argument("--output_dir", type=str, default="output_dir", help="Path to the output directory.")

    parser.add_argument("--group_name", default=None)
    parser.add_argument("--input_wavs_dir", default="LibriTTS")
    parser.add_argument("--input_mels_dir", default="ft_dataset")
    parser.add_argument(
        "--input_training_file", default="tests/LibriTTS/train-full.txt"
    )
    parser.add_argument(
        "--input_validation_file", default="tests/LibriTTS/val-full.txt"
    )

    parser.add_argument(
        "--list_input_unseen_wavs_dir",
        nargs="+",
        default=["tests/LibriTTS", "tests/LibriTTS"],
    )
    parser.add_argument(
        "--list_input_unseen_validation_file",
        nargs="+",
        default=["tests/LibriTTS/dev-clean.txt", "tests/LibriTTS/dev-other.txt"],
    )

    parser.add_argument("--checkpoint_path", default="./logs/bigvgan_finetuned")
    parser.add_argument("--config", default="")

    parser.add_argument("--training_epochs", default=550, type=int)
    parser.add_argument("--stdout_interval", default=100, type=int)
    parser.add_argument("--checkpoint_interval", default=20000, type=int)
    parser.add_argument("--summary_interval", default=1000, type=int)
    parser.add_argument("--validation_interval", default=20000, type=int)

    parser.add_argument(
        "--freeze_step",
        default=0,
        type=int,
        help="freeze D for the first specified steps. G only uses regression loss for these steps.",
    )

    parser.add_argument("--fine_tuning", default=False, type=bool)

    parser.add_argument(
        "--debug",
        action='store_true',
        help="debug mode. skips validation loop throughout training",
    )
    parser.add_argument(
        "--evaluate",
        default=False,
        type=bool,
        help="only run evaluation from checkpoint and exit",
    )
    parser.add_argument(
        "--eval_subsample",
        default=1,
        type=int,
        help="subsampling during evaluation loop",
    )
    parser.add_argument(
        "--skip_seen",
        default=False,
        type=bool,
        help="skip seen dataset. useful for test set inference",
    )
    parser.add_argument(
        "--save_audio",
        action="store_true",
        help="save audio of test set inference to disk",
    )

    args = parser.parse_args()
    with open(args.bigvgan_config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    build_env(args.bigvgan_config, "config.json", args.bigvgan_ckpt_path)

    
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print(f"Batch size per GPU: {h.batch_size}")
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(
            train,
            nprocs=h.num_gpus,
            args=(
                args,
                h,
            ),
        )
    else:
        train(0, args, h)


if __name__ == "__main__":
    main()
