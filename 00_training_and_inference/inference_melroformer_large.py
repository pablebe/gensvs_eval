# Copyright (c) 2025
#   Licensed under the MIT license.

# Adapted from https://github.com/NVIDIA/BigVGAN/tree/main under the MIT license.
#   LICENSE is in incl_licenses directory.

import argparse
import yaml
import time
from ml_collections import ConfigDict
from tqdm import tqdm
import sys
import os
import numpy as np
import glob
import torch
import soundfile as sf
import torch.nn as nn
from baseline_models.util.utils import demix_track, get_model_from_config
from sgmsvs.loudness import calculate_loudness
import warnings
warnings.filterwarnings("ignore")

FADE_LEN = 0.1 # seconds
LOUDNESS_LEVEL = -18 # dBFS

def run_folder(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()
    all_mixtures_path = glob.glob(args.input_folder + '/*.wav')
    total_tracks = len(all_mixtures_path)
    print('Total tracks found: {}'.format(total_tracks))

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    first_chunk_time = None

    for track_number, path in enumerate(all_mixtures_path, 1):
        print(f"\nProcessing track {track_number}/{total_tracks}: {os.path.basename(path)}")

        mix, sr = sf.read(path)
        mixture = torch.tensor(mix.T, dtype=torch.float32)

        if first_chunk_time is not None:
            total_length = mixture.shape[1]
            num_chunks = (total_length + config.inference.chunk_size // config.inference.num_overlap - 1) // (config.inference.chunk_size // config.inference.num_overlap)
            estimated_total_time = first_chunk_time * num_chunks
            print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds")
            sys.stdout.write(f"Estimated time remaining: {estimated_total_time:.2f} seconds\r")
            sys.stdout.flush()

        res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

        for instr in instruments:
            vocals_path = "{}/{}_{}_{}.wav".format(args.store_dir,'separated',instr, os.path.basename(path)[:-4].split('mixture_')[-1])#os.path.basename(path)[:-4])
            if args.output_mono:           
                audio_mono = res[instr].T[:,0]
                fade_in = np.linspace(0, 1, int(FADE_LEN*sr))
                fade_out = np.linspace(1, 0, int(FADE_LEN*sr))
                audio_mono[:int(FADE_LEN*sr)] *= fade_in
                audio_mono[-int(FADE_LEN*sr):] *= fade_out
                audio = np.stack((audio_mono, audio_mono), axis=1)
            else:
                audio  = res[instr].T

            if args.loudness_normalize:
                # Normalize loudness
                L_audio = calculate_loudness(audio, sr)
                L_diff_goal_audio = LOUDNESS_LEVEL - L_audio
                k_scale_audio = 10**(L_diff_goal_audio/20)
                audio = audio * k_scale_audio
            
            sf.write(vocals_path, audio, sr, subtype='FLOAT')

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mel_band_roformer')
    parser.add_argument("--config_path", type=str, help="path to config yaml file")
    parser.add_argument("--model_path", type=str, default='', help="Location of the model")
    parser.add_argument("--input_folder", type=str, help="folder with songs to process")
    parser.add_argument("--store_dir", default="", type=str, help="path to store model outputs")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--output_mono", action="store_true", default=False, help="Whether to output mono audio.")
    parser.add_argument("--loudness_normalize", action="store_true", default=False, help="Whether to normalize the loudness of the output audio.")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
      config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model = get_model_from_config(args.model_type, config)
    if args.model_path != '':
        print('Using model: {}'.format(args.model_path))
        model.load_state_dict(
            torch.load(args.model_path, map_location=torch.device('cpu'))
        )

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if type(device_ids)==int:
            device = torch.device(f'cuda:{device_ids}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not available. Run inference on CPU. It will be very slow...')
        model = model.to(device)

    run_folder(model, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder(None)
