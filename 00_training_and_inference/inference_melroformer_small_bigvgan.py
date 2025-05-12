# Copyright (c) 2025
#   Licensed under the MIT license.

# Adapted from https://github.com/NVIDIA/BigVGAN/tree/main under the MIT license.
#   LICENSE is in incl_licenses directory.

import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
import torch.nn.functional as F
import json
import copy
import numpy as np

from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from os.path import join, dirname
from argparse import ArgumentParser
from librosa import resample
# Set CUDA architecture list
from sgmsvs.sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
from bigvgan_utils.bigvgan import BigVGAN
from baseline_models.MSS_mask_model import MaskingModel
from bigvgan_utils.env import AttrDict
from bigvgan_utils.utils import load_checkpoint
from bigvgan_utils.meldataset import mel_spectrogram
from sgmsvs.loudness import calculate_loudness

SAVE_MELROFORM_AUDIO = False
FADE_LEN = 0.1 # seconds
LOUDNESS_LEVEL = -18 # dBFS

def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--melroformer_ckpt", type=str,  help='Path to model checkpoint')
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    parser.add_argument("--bigvgan_config_file", type=str, default=None, required=True, help="Path to the config file for the BigVGAN model.")
    parser.add_argument("--bigvgan_checkpoint", type=str, default=None, required=True, help="Path to the checkpoint file for the BigVGAN model.")
    parser.add_argument("--bigvgan_use_cuda_kernel", action="store_true", default=False, help="Whether to use the CUDA kernel for the BigVGAN model.")
    parser.add_argument("--output_mono", action="store_true", default=False, help="Whether to output mono audio.")
    parser.add_argument("--loudness_normalize", action="store_true", default=False, help="Whether to normalize the loudness of the output audio.")
    args = parser.parse_args()
    
    if not(torch.cuda.is_available()):
        args.device = torch.device("cpu")
    with open(args.bigvgan_config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    bigvgan_config = AttrDict(json_config)
    bigvgan = BigVGAN(bigvgan_config).to(args.device)
    state_dict_g = load_checkpoint(args.bigvgan_checkpoint, args.device)
    bigvgan.load_state_dict(state_dict_g["generator"])
    bigvgan.eval()
    
    melroform_bigvgan_model = MaskingModel.load_from_checkpoint(args.melroformer_ckpt, map_location=args.device)
    # clone melrformer model
    melroformer = copy.deepcopy(melroform_bigvgan_model.dnn)
    melroformer.eval()
    
    noisy_files = []
    if 'musdb18hq' in args.test_dir:
        noisy_files += sorted(glob.glob(join(args.test_dir, 'mixture.wav')))
        noisy_files += sorted(glob.glob(join(args.test_dir, '**', 'mixture.wav')))
    else:
        noisy_files += sorted(glob.glob(join(args.test_dir, '*.wav')))
        noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.wav')))

    # Check if the model is trained on 48 kHz data
    target_sr = 44100

    # Enhance files
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(args.test_dir, "")
        filename = filename[1:] if filename.startswith("/") else filename

        # Load wav
        y, sr = load(noisy_file)
        num_frames = int(np.ceil(y.shape[1] / bigvgan_config.hop_size))
        target_len = (num_frames ) * bigvgan_config.hop_size
        current_len = y.size(-1)
        pad = max(target_len - current_len, 0)
        if pad != 0:
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')
        # Resample if necessary
        if sr != target_sr:
            y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=target_sr))
        # Backward transform in time domain
        with torch.no_grad():
            norm_fac = y.abs().max()
            y = y / norm_fac
            sep_audio = melroformer(y.unsqueeze(0).to(args.device)).squeeze()
            sep_audio = sep_audio * norm_fac
            mel_sep = mel_spectrogram(
                                        sep_audio.squeeze(),
                                        bigvgan_config.n_fft,
                                        bigvgan_config.num_mels,
                                        bigvgan_config.sampling_rate,
                                        bigvgan_config.hop_size,
                                        bigvgan_config.win_size,
                                        bigvgan_config.fmin,
                                        bigvgan_config.fmax_for_loss,
                                    )
                    
            mel_sep = mel_sep.to(args.device)
            x_hat = bigvgan(mel_sep).squeeze()
            x_hat = x_hat[:,pad//2:-(pad//2+(pad%2))]
            sep_audio = sep_audio[:,pad//2:-(pad//2+(pad%2))]

        if y.shape[0]>1:
            #if stereo put channel dimenion last
            x_hat = x_hat.T
            
        if args.output_mono:           
            audio_mono = x_hat[:,0].cpu()
            fade_in = np.linspace(0, 1, int(FADE_LEN*sr))
            fade_out = np.linspace(1, 0, int(FADE_LEN*sr))
            audio_mono[:int(FADE_LEN*sr)] *= fade_in
            audio_mono[-int(FADE_LEN*sr):] *= fade_out
            x_hat = np.stack((audio_mono, audio_mono), axis=1)
        else:
            x_hat = x_hat.cpu().numpy()
            
        if args.loudness_normalize:
            # Normalize loudness
            L_audio = calculate_loudness(x_hat, sr)
            L_diff_goal_audio = LOUDNESS_LEVEL - L_audio
            k_scale_audio = 10**(L_diff_goal_audio/20)
            x_hat = x_hat * k_scale_audio
        
        # Write enhanced wav file
        filename = 'separated_vocals_'+filename.split('mixture_')[-1]
        makedirs(dirname(join(args.enhanced_dir,filename)), exist_ok=True)
        write(join(args.enhanced_dir, filename), x_hat, target_sr)
        if SAVE_MELROFORM_AUDIO:
            makedirs(dirname(join(os.path.sep.join(args.enhanced_dir.split(os.path.sep)[:-1]),'separated_melroformer_small_from_bigvgan',filename)), exist_ok=True)
            write(join(os.path.sep.join(args.enhanced_dir.split(os.path.sep)[:-1]),'separated_melroformer_small_from_bigvgan', filename), sep_audio.cpu().numpy().T, target_sr)
