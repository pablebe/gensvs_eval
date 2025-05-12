import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
import torch.nn.functional as F
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
from sgmsvs.loudness import calculate_loudness
set_torch_cuda_arch_list()

from baseline_models.MSS_mask_model import MaskingModel 

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
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint')
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    parser.add_argument("--output_mono", action="store_true", default=False, help="Whether to output mono audio.")
    parser.add_argument("--loudness_normalize", action="store_true", default=False, help="Whether to normalize the loudness of the output audio.")

    args = parser.parse_args()
    
    if not(torch.cuda.is_available()):
        args.device = 'cpu'

    model = MaskingModel.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.eval()

    noisy_files = []
    if 'musdb18hq' in args.test_dir:
        noisy_files += sorted(glob.glob(join(args.test_dir, 'mixture.wav')))
        noisy_files += sorted(glob.glob(join(args.test_dir, '**', 'mixture.wav')))
    else:
        noisy_files += sorted(glob.glob(join(args.test_dir, '*.wav')))
        noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.wav')))

    # Check if the model is trained on 48 kHz data
    if model.backbone == 'mel_band_roformer':
        target_sr = 44100
        pad_mode = "reflection"
    elif model.backbone == 'htdemucs':
        target_sr = 44100
        pad_mode = "reflection"

    # Enhance files
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(args.test_dir, "")
        filename = filename[1:] if filename.startswith("/") else filename

        # Load wav
        y, sr = load(noisy_file)
        if model.backbone == 'mel_band_roformer':
            num_frames = int(np.ceil(y.shape[1] / model.dnn.stft_kwargs['hop_length']))
            target_len = (num_frames ) * model.dnn.stft_kwargs['hop_length']
        elif model.backbone == 'htdemucs':
            num_frames = int(np.ceil(y.shape[1] / model.dnn.hop_length))
            target_len = (num_frames ) * model.dnn.hop_length
        else:
            raise ValueError("Unknown model backbone: {}".format(model.backbone))
            
        current_len = y.size(-1)
        pad = max(target_len - current_len, 0)
        if pad != 0:
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')

        y = y.to(args.device)
        norm_factor = y.abs().max()
        y = y/norm_factor
        # Resample if necessary
        if sr != target_sr:
            y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=target_sr))
        # Backward transform in time domain
        with torch.no_grad():
            x_hat = model(y.unsqueeze(0))
        x_hat = x_hat.squeeze().detach()
        # Renormalize
        x_hat = x_hat * norm_factor
        if y.shape[0]>1:
            #if stereo put channel dimenion last
            x_hat = x_hat.T
        x_hat = x_hat[pad//2:-(pad//2+(pad%2)),:]
        
        if args.output_mono:           
            audio_mono = x_hat[:,0].cpu()
            fade_in = np.linspace(0, 1, int(FADE_LEN*sr))
            fade_out = np.linspace(1, 0, int(FADE_LEN*sr))
            audio_mono[:int(FADE_LEN*sr)] *= fade_in
            audio_mono[-int(FADE_LEN*sr):] *= fade_out
            x_hat = np.stack((audio_mono, audio_mono), axis=1)
        else:
            x_hat  = x_hat.cpu().numpy()
            
        if args.loudness_normalize:
            L_audio = calculate_loudness(x_hat, sr)
            L_diff_goal_audio = LOUDNESS_LEVEL - L_audio
            k_scale_audio = 10**(L_diff_goal_audio/20)
            x_hat = x_hat * k_scale_audio
        
        # Write enhanced wav file
        filename = 'separated_vocals_'+filename.split('mixture_')[-1]
        makedirs(dirname(join(args.enhanced_dir,filename)), exist_ok=True)
        write(join(args.enhanced_dir, filename), x_hat, target_sr)
