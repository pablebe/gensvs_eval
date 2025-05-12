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
# Set CUDA architecture list
from sgmsvs.sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()

from sgmsvs.MSS_model import ScoreModel
from sgmsvs.sgmse.util.other import pad_spec
from sgmsvs.loudness import calculate_loudness

CH_BY_CH_PROCESSING = True
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
    parser.add_argument("--sampler_type", type=str, default="pc", help="Sampler type for the PC sampler.")
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=2, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.7, help="SNR value for (annealed) Langevin dynmaics")
    parser.add_argument("--N", type=int, default=35, help="Number of reverse steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
    parser.add_argument("--output_mono", action="store_true", default=False, help="Whether to output mono audio.")
    parser.add_argument("--loudness_normalize", action="store_true", default=False, help="Whether to normalize the loudness of the output audio.")
    
    
    args = parser.parse_args()

    if not(torch.cuda.is_available()):
        args.device = "cpu"

    model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.t_eps = args.t_eps
    model.eval()



    # Get list of noisy files
    noisy_files = []
    if 'musdb18hq' in args.test_dir:
        noisy_files += sorted(glob.glob(join(args.test_dir, 'mixture.wav')))
        noisy_files += sorted(glob.glob(join(args.test_dir, '**', 'mixture.wav')))
    else:
        noisy_files += sorted(glob.glob(join(args.test_dir, '*.wav')))
        noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.wav')))

    # Check if the model is trained on 48 kHz data
    if model.backbone == 'ncsnpp_48k':
        target_sr = 44100#48000
        pad_mode = "reflection"
    elif model.backbone == 'ncsnpp_v2':
        target_sr = 16000
        pad_mode = "reflection"
    else:
        target_sr = 16000
        pad_mode = "zero_pad"

    # Enhance files
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(args.test_dir, "")
        filename = filename[1:] if filename.startswith("/") else filename

        # Load wav
        y, sr = load(noisy_file)

        torch.manual_seed(1234)
        T_orig = y.size(1)

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor
        
        with torch.no_grad():
            # Prepare DNN input
            if y.shape[0]>1:
                Y = torch.unsqueeze(model._forward_transform(model._stft(y.to(args.device))), 1)
            else:
                Y = torch.unsqueeze(model._forward_transform(model._stft(y.to(args.device))), 0)
            Y = pad_spec(Y, mode=pad_mode)

            # Reverse sampling
            if CH_BY_CH_PROCESSING:
                x_hat_ch = []
                for ch in range(Y.shape[0]):
                    if model.sde.__class__.__name__ == 'OUVESDE':
                        if args.sampler_type == 'pc':
                            sampler = model.get_pc_sampler('reverse_diffusion', args.corrector, Y[ch,...][None,...].to(args.device), N=args.N, 
                                corrector_steps=args.corrector_steps, snr=args.snr)
                        elif args.sampler_type == 'ode':
                            sampler = model.get_ode_sampler(Y[ch,...][None,...].to(args.device), N=args.N)
                        else:
                            raise ValueError(f"Sampler type {args.sampler_type} not supported")
                    elif model.sde.__class__.__name__ == 'SBVESDE':
                        sampler_type = 'ode' if args.sampler_type == 'pc' else args.sampler_type
                        sampler = model.get_sb_sampler(sde=model.sde, y=Y[ch,...][None,...].cuda(), sampler_type=sampler_type)
                    else:
                        raise ValueError(f"SDE {model.sde.__class__.__name__} not supported")
                    sample, _ = sampler()
                    
                    # Backward transform in time domain
                    temp = model.to_audio(sample.squeeze(), T_orig)
                    x_hat_ch.append(temp)
                x_hat = torch.stack(x_hat_ch, dim=0)
                
#                print("stop")
        
            else:            
                if model.sde.__class__.__name__ == 'OUVESDE':
                    if args.sampler_type == 'pc':
                        sampler = model.get_pc_sampler('reverse_diffusion', args.corrector, Y.to(args.device), N=args.N, 
                            corrector_steps=args.corrector_steps, snr=args.snr)
                    elif args.sampler_type == 'ode':
                        sampler = model.get_ode_sampler(Y.to(args.device), N=args.N)
                    else:
                        raise ValueError(f"Sampler type {args.sampler_type} not supported")
                elif model.sde.__class__.__name__ == 'SBVESDE':
                    sampler_type = 'ode' if args.sampler_type == 'pc' else args.sampler_type
                    sampler = model.get_sb_sampler(sde=model.sde, y=Y.cuda(), sampler_type=sampler_type)
                else:
                    raise ValueError(f"SDE {model.sde.__class__.__name__} not supported")
                sample, _ = sampler()
                
                # Backward transform in time domain
                x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor
        if y.shape[0]>1:
            #if stereo put channel dimenion last
            x_hat = x_hat.T
            
        if args.output_mono:
            audio_mono = x_hat[:,0].cpu().numpy()
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
