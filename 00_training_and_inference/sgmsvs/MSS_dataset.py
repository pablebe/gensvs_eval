import os
import soundfile#
import string
import numpy as np
import torch
from glob import glob
#import webrtcvad
from torch.utils import data

#TODO: Check how many non-silent stems are in one epoch with 64 samples per track and various percentages of full mix for random mixing and non random mixing

def compute_rms(x_segment,return_stereo=False):
    """ calculate rms value for audio segment """
    #channel dimension has to come last!
    rms = 0.0
    sig_len = 0
    sig_len = x_segment.shape[0]
    rms = np.sum(x_segment**2,0)
    rms = np.sqrt(rms/sig_len)
    if return_stereo:
        return rms
    else:
        return np.mean(rms)
    

def augment_gain(audio: torch.Tensor, low: float = 0.25, high: float = 1.25) -> torch.Tensor:
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def augment_channelswap(audio: torch.Tensor) -> torch.Tensor:
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.tensor(1.0).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def augment_force_stereo(audio: torch.Tensor) -> torch.Tensor:
    # for multichannel > 2, we drop the other channels
    if audio.shape[0] > 2:
        audio = audio[:2, ...]

    if audio.shape[0] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=0)

    return audio



def get_subfoldernames_in_folder(folderPath: string):

    subfolder_list = [f.path.decode("utf-8") for f in os.scandir(folderPath) if f.is_dir()]

    return subfolder_list
class MSSMUSDBDataset(data.Dataset):
    def __init__(self, 
                 foldername_list, 
                 target_str, 
                 random_mix_flag, 
                 augmentation_flag,
                 enforce_full_mix_percentage,
                 duration, 
                 samples_per_track, 
                 valid_flag, 
                 rand_seed=None) -> None:
        super().__init__()

        self.foldername_list = foldername_list
        self.target_str = target_str
        self.random_mixing = random_mix_flag
        self.augmentations = augmentation_flag
        self.full_mix_percentage = enforce_full_mix_percentage
        self.duration = duration
        self.length = len(foldername_list)
        self.valid_flag = valid_flag
        if not(self.valid_flag):
            self.samples_per_track = samples_per_track
        else:
            self.samples_per_track = 1
        self.rand_seed = rand_seed
        #set fs with info of first sample from first track
        self.fs = soundfile.info(os.path.join(foldername_list[0],'mixture.wav')).samplerate
        self.silent_target_counter = 0
        self.silent_mixture_counter = 0
        
        if self.valid_flag:
            self.rms_threshold = 10**(-30/20)
        else:
            self.rms_threshold = 10**(-60/20)

    def __len__(self):
        return self.length*self.samples_per_track

    def get_training_sample(self, 
                            mixture_filepath, 
                            target_bass_filepath, 
                            target_vocals_filepath, 
                            target_other_filepath, 
                            target_drums_filepath, 
                            random_mixing_flag, 
                            augmentation_flag, 
                            duration):

#        if self.valid_flag:
            # set seed for validation set such that always the same samples are extracted from tracks
#            np.random.seed(self.rand_seed) 

        if random_mixing_flag:
            bass_info = soundfile.info(target_bass_filepath)
            seq_dur = int(np.floor(self.fs*duration))

            bass_len = bass_info.frames
            bass_start_index = np.random.randint(np.maximum(bass_len - seq_dur, 0)+1)

            vocals_info = soundfile.info(target_vocals_filepath)
            vocals_len = vocals_info.frames
            vocals_start_index = np.random.randint(np.maximum(vocals_len - seq_dur, 0)+1)

            other_info = soundfile.info(target_other_filepath)
            other_len = other_info.frames
            other_start_index = np.random.randint(np.maximum(other_len - seq_dur, 0)+1)

            drums_info = soundfile.info(target_drums_filepath)
            drums_len = drums_info.frames
            drums_start_index = np.random.randint(np.maximum(drums_len - seq_dur, 0)+1)
     
        else:
                    
            mixture_info = soundfile.info(mixture_filepath)
            mixture_len = mixture_info.frames
            seq_dur = int(np.floor(self.fs*duration))

            mixture_start_index = np.random.randint(np.maximum(mixture_len - seq_dur, 0)+1)
#            mixture, _ = soundfile.read(mixture_filepath, frames = seq_dur, start = mixture_start_index, dtype='float32')

            bass_start_index = mixture_start_index
            vocals_start_index = mixture_start_index
            other_start_index = mixture_start_index
            drums_start_index = mixture_start_index
            
            bass_info = soundfile.info(target_bass_filepath)
            bass_len = bass_info.frames

            vocals_info = soundfile.info(target_vocals_filepath)
            vocals_len = vocals_info.frames            
            
            other_info = soundfile.info(target_other_filepath)
            other_len = other_info.frames

            drums_info = soundfile.info(target_drums_filepath)
            drums_len = drums_info.frames

        target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
        target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
        target_other, _ = soundfile.read(target_other_filepath, frames = seq_dur, start = other_start_index, dtype='float32')
        target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')

        #ensure that the target sources are not silent
        target_bass_rms = compute_rms(target_bass)# np.sqrt(np.mean(target_bass**2))
        target_vocals_rms = compute_rms(target_vocals)#np.sqrt(np.mean(target_vocals**2))
        target_other_rms = compute_rms(target_other)#np.sqrt(np.mean(target_other**2))
        target_drums_rms = compute_rms(target_drums)#np.sqrt(np.mean(target_drums**2))

        if self.target_str=='bass':
            while target_bass_rms <= self.rms_threshold:
                bass_start_index = np.random.randint(np.maximum(bass_len - seq_dur, 0)+1)
                target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
                target_bass_rms = compute_rms(target_bass)
            if not(random_mixing_flag):
                target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
                target_other, _ = soundfile.read(target_other_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
                target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
        elif self.target_str=='vocals':
            loop_ct = 0
            while target_vocals_rms <= self.rms_threshold:
                vocals_start_index = np.random.randint(np.maximum(vocals_len - seq_dur, 0)+1)
                target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
                target_vocals_rms = compute_rms(target_vocals)
                loop_ct += 1
                if loop_ct > 100:
                    print('Target RMS is always below threshold =>lowering threshold to -40dB')
                    old_threshold = self.rms_threshold
                    self.rms_threshold = 10**(-40/20)
            if loop_ct > 100:
                #reset rms threshold for next samples
                self.rms_threshold = old_threshold
            if not(random_mixing_flag):
                target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
                target_other, _ = soundfile.read(target_other_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
                target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
        elif self.target_str=='other':
            while target_other_rms <= self.rms_threshold:
                other_start_index = np.random.randint(np.maximum(other_len - seq_dur, 0)+1)
                target_other, _ = soundfile.read(target_other_filepath, frames = seq_dur, start = other_start_index, dtype='float32')
                target_other_rms = compute_rms(target_other)
            if not(random_mixing_flag):
                target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = other_start_index, dtype='float32')
                target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = other_start_index, dtype='float32')
                target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = other_start_index, dtype='float32')
        elif self.target_str=='drums':
            while target_drums_rms <= self.rms_threshold:
                drums_start_index = np.random.randint(np.maximum(drums_len - seq_dur, 0)+1)
                target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')
                target_drums_rms = compute_rms(target_drums)
            if not(random_mixing_flag):
                target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')
                target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')
                target_other, _ = soundfile.read(target_other_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')

        if np.random.rand() <= self.full_mix_percentage:

            if random_mixing_flag:

                while target_bass_rms <= self.rms_threshold:
                    bass_start_index = np.random.randint(np.maximum(bass_len - seq_dur, 0)+1)
                    target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
                    target_bass_rms = compute_rms(target_bass)# np.sqrt(np.mean(target_bass**2))

                while target_vocals_rms <= self.rms_threshold:
                    vocals_start_index = np.random.randint(np.maximum(vocals_len - seq_dur, 0)+1)
                    target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
                    target_vocals_rms = compute_rms(target_vocals)# np.sqrt(np.mean(target_vocals**2))

                while target_other_rms <= self.rms_threshold:
                    other_start_index = np.random.randint(np.maximum(other_len - seq_dur, 0)+1)
                    target_other, _ = soundfile.read(target_other_filepath, frames = seq_dur, start = other_start_index, dtype='float32')
                    target_other_rms = compute_rms(target_other)# np.sqrt(np.mean(target_vocals**2))

                while target_drums_rms <= self.rms_threshold:
                    drums_start_index = np.random.randint(np.maximum(drums_len - seq_dur, 0)+1)
                    target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')
                    target_drums_rms = compute_rms(target_drums)# np.sqrt(np.mean(target_drums**2))

            else:
                
                rms_index = [target_bass_rms<= self.rms_threshold, target_vocals_rms<= self.rms_threshold, target_other_rms<= self.rms_threshold, target_drums_rms<= self.rms_threshold]

#                target_filepaths = [target_bass_filepath, target_vocals_filepath, target_other_filepaths, target_drums_filepath]

                #if any of the extracted blocks is silent start looking for block in which all stems are not silent
                if any(rms_index):
                    # open full files, segment into seq_dur blocks calculate rms for those blocks. choose random block with rms > -60dB for all target samples
                    target_bass, _ = soundfile.read(target_bass_filepath, dtype='float32')
                    target_drums, _ = soundfile.read(target_drums_filepath, dtype='float32')
                    target_vocals, _ = soundfile.read(target_vocals_filepath, dtype='float32')
                    target_other, _ = soundfile.read(target_other_filepath, dtype='float32')
                    bass_len = len(target_bass)
                    drums_len = len(target_drums)
                    vocals_len = len(target_vocals)
                    other_len = len(target_other)
                    # cut audio to same minimum length of all available stems
                    min_len = np.min([bass_len, drums_len, vocals_len, other_len])
                    target_bass = target_bass[:min_len]
                    target_drums = target_drums[:min_len]
                    target_vocals = target_vocals[:min_len]
                    target_other = target_other[:min_len]

                    #segment stems into seq_dur blocks

                    n_blocks = int(np.floor(min_len/seq_dur))
                    bass_blocks = np.zeros((n_blocks, seq_dur, 2))
                    vocals_blocks = np.zeros((n_blocks, seq_dur, 2))
                    other_blocks = np.zeros((n_blocks, seq_dur, 2))
                    drums_blocks = np.zeros((n_blocks, seq_dur, 2))
                    bass_block_rms = np.zeros(n_blocks)
                    vocals_block_rms = np.zeros(n_blocks)
                    other_block_rms = np.zeros(n_blocks)
                    drums_block_rms = np.zeros(n_blocks)

                    for i in range(n_blocks):
                        bass_blocks[i] = target_bass[i*seq_dur:(i+1)*seq_dur]
                        bass_block_rms[i] = compute_rms(bass_blocks[i])# np.sqrt(np.mean(bass_blocks[i]**2))
                        vocals_blocks[i] = target_vocals[i*seq_dur:(i+1)*seq_dur]
                        vocals_block_rms[i] = compute_rms(vocals_blocks[i])#  np.sqrt(np.mean(vocals_blocks[i]**2))
                        other_blocks[i] = target_other[i*seq_dur:(i+1)*seq_dur]
                        other_block_rms[i] = compute_rms(other_blocks[i])#  np.sqrt(np.mean(other_blocks[i]**2))
                        drums_blocks[i] = target_drums[i*seq_dur:(i+1)*seq_dur]
                        drums_block_rms[i] = compute_rms(drums_blocks[i])#  np.sqrt(np.mean(drums_blocks[i]**2))

                    all_blocks_rms = np.stack((bass_block_rms, vocals_block_rms, other_block_rms, drums_block_rms), axis=1)
                    #determine non-silent blocks
                    rms_logical_index = [bass_block_rms > self.rms_threshold, vocals_block_rms > self.rms_threshold, other_block_rms > self.rms_threshold, drums_block_rms > self.rms_threshold]
                    rms_logical_index = np.array(rms_logical_index)
                    rms_logical_sum = np.sum(rms_logical_index, axis=0)
                    #get rid of blocks with silent target
                    #get rid of blocks with silent target
                    if self.target_str == 'bass':
                        target_rms_idx = bass_block_rms < self.rms_threshold
                    elif self.target_str == 'vocals':
                        target_rms_idx = vocals_block_rms < self.rms_threshold
                    elif self.target_str == 'other':
                        target_rms_idx = other_block_rms < self.rms_threshold
                    elif self.target_str == 'drums':
                        target_rms_idx = drums_block_rms < self.rms_threshold
                    rms_logical_sum[target_rms_idx]=0

                    block_idxs = np.arange(n_blocks)

                    # look for block in which all stems ar non-silent, if there is no block with 4 silents stems look for block with 3 silent stems and so on
                    if any(rms_logical_sum==4):
                        block_idx = np.random.choice(block_idxs[rms_logical_sum==4])
                    elif any(rms_logical_sum==3):
                        block_idx = np.random.choice(block_idxs[rms_logical_sum==3])
                    elif any(rms_logical_sum==2):
                        block_idx = np.random.choice(block_idxs[rms_logical_sum==2])
                    elif any(rms_logical_sum==1):
                        block_idx = np.random.choice(block_idxs[rms_logical_sum==1])
                    
                    target_bass = bass_blocks[block_idx]
                    target_vocals = vocals_blocks[block_idx]
                    target_other = other_blocks[block_idx]
                    target_drums = drums_blocks[block_idx]





        #ensure that the target sources are not silent
        # if self.valid_flag == True:
        #     target_bass_rms = compute_rms(target_bass)#np.sqrt(np.mean(target_bass**2))
        #     target_vocals_rms = compute_rms(target_vocals)#np.sqrt(np.mean(target_vocals**2))
        #     target_other_rms = compute_rms(target_other)#np.sqrt(np.mean(target_other**2))
        #     target_drums_rms = compute_rms(target_drums)#np.sqrt(np.mean(target_drums**2))

        #     if random_mixing_flag:
        #         if self.target_str=='bass':
        #             while target_bass_rms <= self.rms_threshold:
        #                 bass_start_index = np.random.randint(np.maximum(bass_len - seq_dur, 0)+1)
        #                 target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
        #                 target_bass_rms = compute_rms(target_bass)#np.sqrt(np.mean(target_bass**2))

        #         elif self.target_str=='vocals':
        #             while target_vocals_rms <= self.rms_threshold:
        #                 vocals_start_index = np.random.randint(np.maximum(vocals_len - seq_dur, 0)+1)
        #                 target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
        #                 target_vocals_rms = compute_rms(target_vocals)#np.sqrt(np.mean(target_vocals**2))

        #         elif self.target_str=='other':
        #             while target_other_rms <= 10**(-60/20): 
        #                 other_start_index = np.random.randint(np.maximum(other_len - seq_dur, 0)+1)
        #                 target_other, _ = soundfile.read(target_other_filepath, frames = seq_dur, start = other_start_index, dtype='float32')
        #                 target_other_rms = compute_rms(target_other)#np.sqrt(np.mean(target_other**2))
                
        #         elif self.target_str=='drums':
        #             while target_drums_rms <= 10**(-60/20):
        #                 drums_start_index = np.random.randint(np.maximum(drums_len - seq_dur, 0)+1)
        #                 target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')
        #                 target_drums_rms = compute_rms(target_drums)#np.sqrt(np.mean(target_drums**2))

        #     else:




                # if self.target_str=='bass':
                #     while target_bass_rms <= 10**(-60/20):
                #         mixture_start_index = np.random.randint(np.maximum(mixture_len - seq_dur, 0)+1)
                #         target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = mixture_start_index, dtype='float32')
                #         target_bass_rms = compute_rms(target_bass)#np.sqrt(np.mean(target_bass**2))

                # elif self.target_str=='vocals':
                #     while target_vocals_rms <= 10**(-60/20):
                #         mixture_start_index = np.random.randint(np.maximum(mixture_len - seq_dur, 0)+1)
                #         target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = mixture_start_index, dtype='float32')
                #         target_vocals_rms = compute_rms(target_vocals)# np.sqrt(np.mean(target_vocals**2))

                # elif self.target_str=='other':
                #     while target_other_rms <= 10**(-60/20):
                #         mixture_start_index = np.random.randint(np.maximum(mixture_len - seq_dur, 0)+1)
                #         target_other, _ = soundfile.read(target_other_filepath, frames = seq_dur, start = mixture_start_index, dtype='float32')
                #         target_other_rms = compute_rms(target_other)# np.sqrt(np.mean(target_other**2))

                # elif self.target_str=='drums':
                #     while target_drums_rms <= 10**(-60/20):
                #         mixture_start_index = np.random.randint(np.maximum(mixture_len - seq_dur, 0)+1)
                #         target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = mixture_start_index, dtype='float32')
                #         target_drums_rms = compute_rms(target_drums)# np.sqrt(np.mean(target_drums**2))


        else: 
            # still ensure that mixture is not silent
            temp_mixture = target_bass+target_vocals+target_other+target_drums#torch.stack((target_bass, target_vocals, target_other, target_drums),-1).sum(-1)
            mixture_rms = compute_rms(temp_mixture)
            while mixture_rms <= self.rms_threshold:
                if random_mixing_flag:
                    bass_info = soundfile.info(target_bass_filepath)
                    seq_dur = int(np.floor(self.fs*duration))

                    bass_len = bass_info.frames
                    bass_start_index = np.random.randint(np.maximum(bass_len - seq_dur, 0)+1)

                    vocals_info = soundfile.info(target_vocals_filepath)
                    vocals_len = vocals_info.frames
                    vocals_start_index = np.random.randint(np.maximum(vocals_len - seq_dur, 0)+1)

                    other_info = soundfile.info(target_other_filepath)
                    other_len = other_info.frames
                    other_start_index = np.random.randint(np.maximum(other_len - seq_dur, 0)+1)

                    drums_info = soundfile.info(target_drums_filepath)
                    drums_len = drums_info.frames
                    drums_start_index = np.random.randint(np.maximum(drums_len - seq_dur, 0)+1)
            
                else:
                            
                    mixture_info = soundfile.info(mixture_filepath)
                    mixture_len = mixture_info.frames
                    seq_dur = int(np.floor(self.fs*duration))

                    mixture_start_index = np.random.randint(np.maximum(mixture_len - seq_dur, 0)+1)
        #            mixture, _ = soundfile.read(mixture_filepath, frames = seq_dur, start = mixture_start_index, dtype='float32')

                    bass_start_index = mixture_start_index
                    vocals_start_index = mixture_start_index
                    other_start_index = mixture_start_index
                    drums_start_index = mixture_start_index
                    

                target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
                target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
                target_other, _ = soundfile.read(target_other_filepath, frames = seq_dur, start = other_start_index, dtype='float32')
                target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')

                temp_mixture = target_bass+target_vocals+target_other+target_drums#torch.stack((target_bass, target_vocals, target_other, target_drums),-1).sum(-1)
                mixture_rms = compute_rms(temp_mixture)


        if augmentation_flag:
            audio_bass = torch.as_tensor(target_bass.T, dtype=torch.float32)
            audio_bass = augment_force_stereo(audio_bass)
            audio_bass = augment_gain(audio_bass)
            audio_bass = augment_channelswap(audio_bass)
            target_bass = audio_bass

            audio_vocals = torch.as_tensor(target_vocals.T, dtype=torch.float32)
            audio_vocals = augment_force_stereo(audio_vocals)
            audio_vocals = augment_gain(audio_vocals)
            audio_vocals = augment_channelswap(audio_vocals)
            target_vocals = audio_vocals

            audio_other = torch.as_tensor(target_other.T, dtype=torch.float32)
            audio_other = augment_force_stereo(audio_other)
            audio_other = augment_gain(audio_other)
            audio_other = augment_channelswap(audio_other)
            target_other = audio_other

            audio_drums = torch.as_tensor(target_drums.T, dtype=torch.float32)
            audio_drums = augment_force_stereo(audio_drums)
            audio_drums = augment_gain(audio_drums)
            audio_drums = augment_channelswap(audio_drums)
            target_drums = audio_drums

        else:
            target_bass = torch.Tensor(target_bass.T)
            target_vocals = torch.Tensor(target_vocals.T)
            target_other = torch.Tensor(target_other.T)
            target_drums = torch.Tensor(target_drums.T)

        mixture = torch.stack((target_bass, target_vocals, target_other, target_drums),-1).sum(-1)

#        if self.valid_flag:
#            np.random.seed() #reset seed after validation set as been created

        if self.target_str == 'bass':
            target = target_bass
        elif self.target_str == 'vocals':
            target = target_vocals
        elif self.target_str == 'other':
            target = target_other
        elif self.target_str == 'drums':
            target = target_drums

        target_rms = compute_rms(target.numpy().T, return_stereo=True)
        target_rms[0] = max((target_rms[0], self.rms_threshold))
        target_rms[1] = max((target_rms[1], self.rms_threshold))


        if (torch.Tensor(mixture)==0).all():
            print('mixture is silent')
            self.silent_mixture_counter += 1
#            breakpoint()

        if (torch.Tensor(target)==0).all():
            print('target is silent')
            self.silent_target_counter += 1
#            breakpoint()

        return torch.Tensor(mixture), torch.Tensor(target), target_rms#, torch.Tensor(target_bass), torch.Tensor(target_vocals), torch.Tensor(target_other), torch.Tensor(target_drums), vocal_rms#, mixture_filepath#, mixture_start_index#x, y_bass, y_vocals, y_other, y_drums
    
    def update_samples_per_track(self, samples_per_track_new):
        self.samples_per_track = samples_per_track_new

    def __getitem__(self, item):
        mixture_track_folder = self.foldername_list[item//self.samples_per_track]

        if self.random_mixing:      
            item_bass = np.random.randint(self.__len__())//self.samples_per_track
            while item_bass == item//self.samples_per_track:
                item_bass = np.random.randint(self.__len__())//self.samples_per_track

            item_vocals = np.random.randint(self.__len__())//self.samples_per_track
            while item_vocals == item//self.samples_per_track:
                item_vocals = np.random.randint(self.__len__())//self.samples_per_track

            item_other = np.random.randint(self.__len__())//self.samples_per_track
            while item_other == item//self.samples_per_track:
                item_other = np.random.randint(self.__len__())//self.samples_per_track

            item_drums = np.random.randint(self.__len__())//self.samples_per_track
            while item_drums == item//self.samples_per_track:
                item_drums = np.random.randint(self.__len__())//self.samples_per_track
             
            track_folder_bass = self.foldername_list[item_bass]
            track_folder_vocals = self.foldername_list[item_vocals]
            track_folder_other = self.foldername_list[item_other]
            track_folder_drums = self.foldername_list[item_drums]
        else:
            track_folder_bass = self.foldername_list[item//self.samples_per_track]
            track_folder_vocals =self.foldername_list[item//self.samples_per_track]
            track_folder_other = self.foldername_list[item//self.samples_per_track]
            track_folder_drums = self.foldername_list[item//self.samples_per_track]

        mixture_filepath = os.path.join(mixture_track_folder,'mixture.wav')
        target_bass_filepath = os.path.join(track_folder_bass, 'bass.wav')
        target_vocals_filepath = os.path.join(track_folder_vocals,'vocals.wav')
        target_other_filepath = os.path.join(track_folder_other,'other.wav')
        target_drums_filepath = os.path.join(track_folder_drums,'drums.wav')

        if self.valid_flag:
            np.random.seed(item+self.rand_seed)
            temp = self.get_training_sample(mixture_filepath, target_bass_filepath, target_vocals_filepath, target_other_filepath, target_drums_filepath, self.random_mixing, self.augmentations, self.duration)
            np.random.seed()
        else:
            temp = self.get_training_sample(mixture_filepath, target_bass_filepath, target_vocals_filepath, target_other_filepath, target_drums_filepath, self.random_mixing, self.augmentations, self.duration)

        
        return temp
    


class MSSMoisesDBDataset(data.Dataset):
    def __init__(self, 
                 foldername_list, 
                 target_str, 
                 random_mix_flag, 
                 augmentation_flag, 
                 duration, 
                 samples_per_track,
                 valid_flag,
                 enforce_full_mix_percentage, 
                 rand_seed=None) -> None:
        
        super().__init__()
        #remove folders that do not contain the target source
        temp_list = np.array(foldername_list)
        for folder in np.array(foldername_list):
            subfolders = get_subfoldernames_in_folder(folder)
            contains_target = [True if target_str in subfolder else False for subfolder in subfolders]
            if not any(contains_target):
                temp_list = np.delete(temp_list, temp_list==folder)
        foldername_list = temp_list
        #check if other sources are present in folders
        for folder in foldername_list:
            subfolders = get_subfoldernames_in_folder(folder)
            contains_bass = [True if 'bass' in subfolder else False for subfolder in subfolders]
            contains_drums = [True if 'drums' in subfolder else False for subfolder in subfolders]
            if not any(contains_bass) or not any(contains_drums):
                temp_list = np.delete(temp_list,temp_list==folder)
        foldername_list = temp_list

        self.foldername_list = foldername_list
        self.target_str = target_str
        self.random_mixing = random_mix_flag
        self.augmentations = augmentation_flag
        self.duration = duration
        self.length = len(foldername_list)
#        self.samples_per_track = samples_per_track
        self.valid_flag = valid_flag
        if not(self.valid_flag):
            self.samples_per_track = samples_per_track
        else:
            self.samples_per_track = 1
        self.full_mix_percentage = enforce_full_mix_percentage
        self.rand_seed = rand_seed
        #set fs with info of first sample from first track
        temp_folder = get_subfoldernames_in_folder(foldername_list[0])
        temp_wav = []
        temp_wav += sorted(glob(os.path.join(temp_folder[0], '*.wav')))
        temp_wav = temp_wav[0]
        self.fs = soundfile.info(temp_wav).samplerate
        self.silent_target_counter = 0
        self.silent_mixture_counter = 0
        
        if self.valid_flag:
            self.rms_threshold = 10**(-30/20)
        else:
            self.rms_threshold = 10**(-60/20)

    def __len__(self):
        return self.length*self.samples_per_track

    def get_training_sample(self, 
                            target_bass_folder, 
                            target_vocals_folder, 
                            target_other_folders, 
                            target_drums_folder, 
                            random_mixing_flag, 
                            augmentation_flag, 
                            duration):
        # TODO: Incorporate the valid flag in the same way as in the MUSDB dataset class!
#        if self.valid_flag:
            # set seed for validation set such that always the same samples are extracted from tracks
#            np.random.seed(self.rand_seed) 

        target_bass_filepath = [os.path.join(target_bass_folder,'bass',f) if f.endswith('.wav') else None for f in os.listdir(os.path.join(target_bass_folder,'bass'))][0]
        target_vocals_filepath = [os.path.join(target_vocals_folder,'vocals',f) if f.endswith('.wav') else None for f in os.listdir(os.path.join(target_vocals_folder,'vocals'))][0]
        target_drums_filepath = [os.path.join(target_drums_folder,'drums',f) if f.endswith('.wav') else None for f in os.listdir(os.path.join(target_drums_folder,'drums'))][0]
        target_other_filepaths = [os.path.join(f, os.listdir(f)[0].decode('utf-8')) for f in target_other_folders if os.listdir(f)[0].decode('utf-8').endswith('.wav')]
        if random_mixing_flag:
            bass_info = soundfile.info(target_bass_filepath)
            bass_len = bass_info.frames
            seq_dur = int(np.floor(self.fs*duration))

            bass_start_index = np.random.randint(np.maximum(bass_len - seq_dur, 0)+1)

            vocals_info = soundfile.info(target_vocals_filepath)
            vocals_len = vocals_info.frames
            vocals_start_index = np.random.randint(np.maximum(vocals_len - seq_dur, 0)+1)

            other_start_index = []
            other_lens = []
            for file_other in target_other_filepaths:
                other_info = soundfile.info(file_other)
                other_len = other_info.frames
                other_lens.append(other_len)
                other_start_index.append(np.random.randint(np.maximum(other_len - seq_dur, 0)+1))
            # other_info = soundfile.info(target_other_filepaths[0])
            # other_len = other_info.frames
            # other_start_index = np.random.randint(np.maximum(other_len - seq_dur, 0)+1)

            drums_info = soundfile.info(target_drums_filepath)
            drums_len = drums_info.frames
            drums_start_index = np.random.randint(np.maximum(drums_len - seq_dur, 0)+1)


            target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
            target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
            target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')

            # sum other stem
            if random_mixing_flag:
                target_other = np.zeros((seq_dur,2))
                for file_other, start_idx in zip(target_other_filepaths, other_start_index):
                    temp_other, _ = soundfile.read(file_other, frames = seq_dur, start = start_idx, dtype='float32')
                    target_other += temp_other        
            else:    
                target_other = np.zeros((seq_dur,2))
                for file_other in target_other_filepaths:
                    temp_other, _ = soundfile.read(file_other, frames = seq_dur, start = other_start_index, dtype='float32')
                    target_other += temp_other


            target_bass_rms = compute_rms(target_bass)# np.sqrt(np.mean(target_bass**2))
            target_vocals_rms = compute_rms(target_vocals)#np.sqrt(np.mean(target_vocals**2))
            target_other_rms = compute_rms(target_other)#np.sqrt(np.mean(target_other**2))
            target_drums_rms = compute_rms(target_drums)#np.sqrt(np.mean(target_drums**2))


            #ensure target is not silent
            if self.target_str=='bass':
                while target_bass_rms <= self.rms_threshold:
                    bass_start_index = np.random.randint(np.maximum(bass_len - seq_dur, 0)+1)
                    target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
                    target_bass_rms = compute_rms(target_bass)
            elif self.target_str=='vocals':
                while target_vocals_rms <= self.rms_threshold:
                    vocals_start_index = np.random.randint(np.maximum(vocals_len - seq_dur, 0)+1)
                    target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
                    target_vocals_rms = compute_rms(target_vocals)
            elif self.target_str=='other':
                while target_other_rms <= self.rms_threshold: 
                    other_start_idex = []
                    for file_other in target_other_filepaths:
                        other_info = soundfile.info(file_other)
                        other_len = other_info.frames
                        other_start_idex.append(np.random.randint(np.maximum(other_len - seq_dur, 0)+1))  
                    target_other = np.zeros((seq_dur,2))

                    for file_other, start_idx in zip(target_other_filepaths, other_start_idex):
                        temp_other, _ = soundfile.read(file_other, frames = seq_dur, start = start_idx, dtype='float32')
                        target_other += temp_other                        
                    target_other_rms = compute_rms(target_other)# np.sqrt(np.mean(target_other**2))
            elif self.target_str=='drums':
                while target_drums_rms <= self.rms_threshold:
                    drums_start_index = np.random.randint(np.maximum(drums_len - seq_dur, 0)+1)
                    target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')
                    target_drums_rms = compute_rms(target_drums)

        else:
            conds = [True, True, True, True]
            while any(conds):
                bass_info = soundfile.info(target_bass_filepath)
                bass_len = bass_info.frames
                seq_dur = int(np.floor(self.fs*duration))
                mixture_start_index = np.random.randint(np.maximum(bass_len - seq_dur, 0)+1)
    #            mixture, _ = soundfile.read(mixture_filepath, frames = seq_dur, start = mixture_start_index, dtype='float32')

                vocals_info = soundfile.info(target_vocals_filepath)
                vocals_len = vocals_info.frames

                drums_info = soundfile.info(target_drums_filepath)
                drums_len = drums_info.frames

    #            bass_start_index = mixture_start_index
    #            vocals_start_index = mixture_start_index
    #            other_start_index = mixture_start_index
    #            drums_start_index = mixture_start_index

    #            other_start_index = mixture_start_index
                other_lens = []
                other_conds = []
                for file_other in target_other_filepaths:
                    other_info = soundfile.info(file_other)
                    other_len = other_info.frames
                    other_lens.append(other_len)
                    other_conds.append(not((mixture_start_index + seq_dur) < other_len))
    #               if not((mixture_start_index + seq_dur) < other_len):
    #                   breakpoint()
    #               assert ((mixture_start_index + seq_dur) < other_len), "Other sample is too short for the desired duration!"

                len_conds = [not((mixture_start_index + seq_dur) < bass_len), not((mixture_start_index + seq_dur) < vocals_len), not((mixture_start_index + seq_dur) < drums_len)]
                len_conds = len_conds + other_conds

                other_start_index = mixture_start_index
                bass_start_index = mixture_start_index
                vocals_start_index = mixture_start_index
                drums_start_index = mixture_start_index

                target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
                target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
                target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')

                # sum other stem
                if random_mixing_flag:
                    target_other = np.zeros((seq_dur,2))
                    for file_other, start_idx in zip(target_other_filepaths, other_start_index):
                        temp_other, _ = soundfile.read(file_other, frames = seq_dur, start = start_idx, dtype='float32')
                        target_other += temp_other        
                else:    
                    target_other = np.zeros((seq_dur,2))
                    for file_other in target_other_filepaths:
                        temp_other, _ = soundfile.read(file_other, frames = seq_dur, start = other_start_index, dtype='float32')
                        target_other += temp_other


                target_bass_rms = compute_rms(target_bass)# np.sqrt(np.mean(target_bass**2))
                target_vocals_rms = compute_rms(target_vocals)#np.sqrt(np.mean(target_vocals**2))
                target_other_rms = compute_rms(target_other)#np.sqrt(np.mean(target_other**2))
                target_drums_rms = compute_rms(target_drums)#np.sqrt(np.mean(target_drums**2))
                                 
                #ensure target is not silent
                if self.target_str=='bass':
                        conds = conds + [target_bass_rms <= self.rms_threshold]
                elif self.target_str=='vocals':
                        conds = conds + [target_vocals_rms <= self.rms_threshold]
                elif self.target_str=='other':
                        conds = conds + [target_other_rms <= self.rms_threshold]
                elif self.target_str=='drums':
                        conds = conds + [target_drums_rms <= self.rms_threshold]

#                print("stop")
#               other_start_index.append(mixture_start_index)



        # target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
        # target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
        # target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')

        # # sum other stem
        # if random_mixing_flag:
        #     target_other = np.zeros((seq_dur,2))
        #     for file_other, start_idx in zip(target_other_filepaths, other_start_index):
        #         temp_other, _ = soundfile.read(file_other, frames = seq_dur, start = start_idx, dtype='float32')
        #         target_other += temp_other        
        # else:    
        #     target_other = np.zeros((seq_dur,2))
        #     for file_other in target_other_filepaths:
        #         temp_other, _ = soundfile.read(file_other, frames = seq_dur, start = other_start_index, dtype='float32')
        #         target_other += temp_other


        # target_bass_rms = compute_rms(target_bass)# np.sqrt(np.mean(target_bass**2))
        # target_vocals_rms = compute_rms(target_vocals)#np.sqrt(np.mean(target_vocals**2))
        # target_other_rms = compute_rms(target_other)#np.sqrt(np.mean(target_other**2))
        # target_drums_rms = compute_rms(target_drums)#np.sqrt(np.mean(target_drums**2))



        #ensure that the target sources are not silent
        if np.random.rand() <= self.full_mix_percentage:
            if random_mixing_flag:
                # for random mixing => full mixture is always given here because non silent samples are discarded.
                while target_bass_rms <= self.rms_threshold:
                    bass_start_index = np.random.randint(np.maximum(bass_len - seq_dur, 0)+1)
                    target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
                    target_bass_rms = compute_rms(target_bass)# np.sqrt(np.mean(target_bass**2))

                while target_vocals_rms <= self.rms_threshold:
                    vocals_start_index = np.random.randint(np.maximum(vocals_len - seq_dur, 0)+1)
                    target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
                    target_vocals_rms = compute_rms(target_vocals)# np.sqrt(np.mean(target_vocals**2))

                while target_other_rms <= self.rms_threshold: 
                    other_start_idex = []
                    for file_other in target_other_filepaths:
                        other_info = soundfile.info(file_other)
                        other_len = other_info.frames
                        other_start_idex.append(np.random.randint(np.maximum(other_len - seq_dur, 0)+1))  
                    target_other = np.zeros((seq_dur,2))

                    for file_other, start_idx in zip(target_other_filepaths, other_start_idex):
                        temp_other, _ = soundfile.read(file_other, frames = seq_dur, start = start_idx, dtype='float32')
                        target_other += temp_other                        
                    target_other_rms = compute_rms(target_other)# np.sqrt(np.mean(target_other**2))
            
                while target_drums_rms <= self.rms_threshold:
                    drums_start_index = np.random.randint(np.maximum(drums_len - seq_dur, 0)+1)
                    target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')
                    target_drums_rms = compute_rms(target_drums)# np.sqrt(np.mean(target_drums**2))

            else:
                
                rms_index = [target_bass_rms<= self.rms_threshold, target_vocals_rms<= self.rms_threshold, target_other_rms<= self.rms_threshold, target_drums_rms<= self.rms_threshold]

                target_filepaths = [target_bass_filepath, target_vocals_filepath, target_other_filepaths, target_drums_filepath]

                #if any of the extracted blocks is silent start looking for block in which all stems are not silent
                if any(rms_index):
                    # open full files, segment into seq_dur blocks calculate rms for those blocks. choose random block with rms > -60dB for all target samples
                    target_bass, _ = soundfile.read(target_bass_filepath, dtype='float32')
                    target_drums, _ = soundfile.read(target_drums_filepath, dtype='float32')
                    target_vocals, _ = soundfile.read(target_vocals_filepath, dtype='float32')
                    bass_len = len(target_bass)
                    drums_len = len(target_drums)
                    vocals_len = len(target_vocals)
                    # cut audio to same minimum length of all available stems
                    min_len = np.min(np.concatenate(([bass_len, drums_len, vocals_len], other_lens)))
                    target_bass = target_bass[:min_len]
                    target_drums = target_drums[:min_len]
                    target_vocals = target_vocals[:min_len]

                    target_other = np.zeros((min_len,2))
                    for file_other, sig_len in zip(target_other_filepaths,other_lens):
                        temp_other, _ = soundfile.read(file_other, frames=min_len, dtype='float32')
                        target_other += temp_other

                    #segment stems into seq_dur blocks

                    n_blocks = int(np.floor(min_len/seq_dur))
                    bass_blocks = np.zeros((n_blocks, seq_dur, 2))
                    vocals_blocks = np.zeros((n_blocks, seq_dur, 2))
                    other_blocks = np.zeros((n_blocks, seq_dur, 2))
                    drums_blocks = np.zeros((n_blocks, seq_dur, 2))
                    bass_block_rms = np.zeros(n_blocks)
                    vocals_block_rms = np.zeros(n_blocks)
                    other_block_rms = np.zeros(n_blocks)
                    drums_block_rms = np.zeros(n_blocks)

                    for i in range(n_blocks):
                        bass_blocks[i] = target_bass[i*seq_dur:(i+1)*seq_dur]
                        bass_block_rms[i] = compute_rms(bass_blocks[i])# np.sqrt(np.mean(bass_blocks[i]**2))
                        vocals_blocks[i] = target_vocals[i*seq_dur:(i+1)*seq_dur]
                        vocals_block_rms[i] = compute_rms(vocals_blocks[i])#  np.sqrt(np.mean(vocals_blocks[i]**2))
                        other_blocks[i] = target_other[i*seq_dur:(i+1)*seq_dur]
                        other_block_rms[i] = compute_rms(other_blocks[i])#  np.sqrt(np.mean(other_blocks[i]**2))
                        drums_blocks[i] = target_drums[i*seq_dur:(i+1)*seq_dur]
                        drums_block_rms[i] = compute_rms(drums_blocks[i])#  np.sqrt(np.mean(drums_blocks[i]**2))

                    all_blocks_rms = np.stack((bass_block_rms, vocals_block_rms, other_block_rms, drums_block_rms), axis=1)
                    #determine non-silent blocks
                    rms_logical_index = [bass_block_rms > self.rms_threshold, vocals_block_rms > self.rms_threshold, other_block_rms > self.rms_threshold, drums_block_rms > self.rms_threshold]
                    rms_logical_index = np.array(rms_logical_index)
                    rms_logical_sum = np.sum(rms_logical_index, axis=0)
                    #get rid of blocks with silent target
                    if self.target_str == 'bass':
                        target_rms_idx = bass_block_rms < self.rms_threshold
                    elif self.target_str == 'vocals':
                        target_rms_idx = vocals_block_rms < self.rms_threshold
                    elif self.target_str == 'other':
                        target_rms_idx = other_block_rms < self.rms_threshold
                    elif self.target_str == 'drums':
                        target_rms_idx = drums_block_rms < self.rms_threshold
                    rms_logical_sum[target_rms_idx]=0

                    block_idxs = np.arange(n_blocks)

                    # look for block in which all stems ar non-silent, if there is no block with 4 silents stems look for block with 3 silent stems and so on
                    if any(rms_logical_sum==4):
                        block_idx = np.random.choice(block_idxs[rms_logical_sum==4])
                    elif any(rms_logical_sum==3):
                        block_idx = np.random.choice(block_idxs[rms_logical_sum==3])
                    elif any(rms_logical_sum==2):
                        block_idx = np.random.choice(block_idxs[rms_logical_sum==2])
                    elif any(rms_logical_sum==1):
                        block_idx = np.random.choice(block_idxs[rms_logical_sum==1])
                    else:
                        AssertionError('Level of all stems in signal are too silent!')

                    target_bass = bass_blocks[block_idx]
                    target_vocals = vocals_blocks[block_idx]
                    target_other = other_blocks[block_idx]
                    target_drums = drums_blocks[block_idx]

        else:
            # still ensure that mixture is not silent
            temp_mixture = target_bass+target_vocals+target_other+target_drums
            mixture_rms = compute_rms(temp_mixture)
            while mixture_rms <= self.rms_threshold:
                if random_mixing_flag:
                    bass_info = soundfile.info(target_bass_filepath)
                    bass_len = bass_info.frames
                    seq_dur = int(np.floor(self.fs*duration))

                    bass_start_index = np.random.randint(np.maximum(bass_len - seq_dur, 0)+1)

                    vocals_info = soundfile.info(target_vocals_filepath)
                    vocals_len = vocals_info.frames
                    vocals_start_index = np.random.randint(np.maximum(vocals_len - seq_dur, 0)+1)

                    other_start_index = []
                    other_lens = []
                    for file_other in target_other_filepaths:
                        other_info = soundfile.info(file_other)
                        other_len = other_info.frames
                        other_lens.append(other_len)
                        other_start_index.append(np.random.randint(np.maximum(other_len - seq_dur, 0)+1))


                    drums_info = soundfile.info(target_drums_filepath)
                    drums_len = drums_info.frames
                    drums_start_index = np.random.randint(np.maximum(drums_len - seq_dur, 0)+1)
            
                else:
                            
                    mixture_info = soundfile.info(target_bass_filepath)
                    mixture_len = mixture_info.frames
                    seq_dur = int(np.floor(self.fs*duration))
                    mixture_start_index = np.random.randint(np.maximum(mixture_len - seq_dur, 0)+1)

                    bass_start_index = mixture_start_index
                    vocals_start_index = mixture_start_index
                    drums_start_index = mixture_start_index

                    other_start_index = mixture_start_index
                    other_lens = []
                    for file_other in target_other_filepaths:
                        other_info = soundfile.info(file_other)
                        other_len = other_info.frames
                        other_lens.append(other_len)
                        assert ((other_start_index + seq_dur) < other_len), "Other sample is too short for the desired duration!"
      
                target_bass, _ = soundfile.read(target_bass_filepath, frames = seq_dur, start = bass_start_index, dtype='float32')
                target_vocals, _ = soundfile.read(target_vocals_filepath, frames = seq_dur, start = vocals_start_index, dtype='float32')
                target_drums, _ = soundfile.read(target_drums_filepath, frames = seq_dur, start = drums_start_index, dtype='float32')

                # sum other stem
                if random_mixing_flag:
                    target_other = np.zeros((seq_dur,2))
                    for file_other, start_idx in zip(target_other_filepaths, other_start_index):
                        temp_other, _ = soundfile.read(file_other, frames = seq_dur, start = start_idx, dtype='float32')
                        target_other += temp_other        
                else:    
                    target_other = np.zeros((seq_dur,2))
                    for file_other in target_other_filepaths:
                        temp_other, _ = soundfile.read(file_other, frames = seq_dur, start = other_start_index, dtype='float32')
                        target_other += temp_other

                temp_mixture = target_bass+target_vocals+target_other+target_drums
                mixture_rms = compute_rms(temp_mixture)

        if augmentation_flag:
            audio_bass = torch.as_tensor(target_bass.T, dtype=torch.float32)
            audio_bass = augment_force_stereo(audio_bass)
            audio_bass = augment_gain(audio_bass)
            audio_bass = augment_channelswap(audio_bass)
            target_bass = audio_bass

            audio_vocals = torch.as_tensor(target_vocals.T, dtype=torch.float32)
            audio_vocals = augment_force_stereo(audio_vocals)
            audio_vocals = augment_gain(audio_vocals)
            audio_vocals = augment_channelswap(audio_vocals)
            target_vocals = audio_vocals

            audio_other = torch.as_tensor(target_other.T, dtype=torch.float32)
            audio_other = augment_force_stereo(audio_other)
            audio_other = augment_gain(audio_other)
            audio_other = augment_channelswap(audio_other)
            target_other = audio_other

            audio_drums = torch.as_tensor(target_drums.T, dtype=torch.float32)
            audio_drums = augment_force_stereo(audio_drums)
            audio_drums = augment_gain(audio_drums)
            audio_drums = augment_channelswap(audio_drums)
            target_drums = audio_drums

        else:
            target_bass = torch.Tensor(target_bass.T)
            target_vocals = torch.Tensor(target_vocals.T)
            target_other = torch.Tensor(target_other.T)
            target_drums = torch.Tensor(target_drums.T)

        mixture = torch.stack((target_bass, target_vocals, target_other, target_drums),-1).sum(-1)

        if self.target_str == 'bass':
            target = target_bass
        elif self.target_str == 'vocals':
            target = target_vocals
        elif self.target_str == 'other':
            target = target_other
        elif self.target_str == 'drums':
            target = target_drums

        target_rms = compute_rms(target.numpy().T, return_stereo=True)
        target_rms[0] = max((target_rms[0], self.rms_threshold))
        target_rms[1] = max((target_rms[1], self.rms_threshold))

        if (torch.Tensor(mixture)==0).all():
            print('mixture is silent')
            self.silent_mixture_counter += 1

        if (torch.Tensor(target)==0).all():
            print('target is silent')
            self.silent_target_counter += 1

        return torch.Tensor(mixture), torch.Tensor(target), target_rms#, torch.Tensor(target_bass), torch.Tensor(target_vocals), torch.Tensor(target_other), torch.Tensor(target_drums), vocal_rms#, mixture_start_index#x, y_bass, y_vocals, y_other, y_drums

    def update_samples_per_track(self, samples_per_track_new):
        self.samples_per_track = samples_per_track_new

    def __getitem__(self, item):
#        mixture_track_folder = self.foldername_list[item//self.samples_per_track]

        if self.random_mixing:      
            item_bass = np.random.randint(self.__len__())//self.samples_per_track
            while item_bass == item//self.samples_per_track:
                item_bass = np.random.randint(self.__len__())//self.samples_per_track

            item_vocals = np.random.randint(self.__len__())//self.samples_per_track
            while item_vocals == item//self.samples_per_track:
                item_vocals = np.random.randint(self.__len__())//self.samples_per_track

            item_other = np.random.randint(self.__len__())//self.samples_per_track
            while item_other == item//self.samples_per_track:
                item_other = np.random.randint(self.__len__())//self.samples_per_track

            item_drums = np.random.randint(self.__len__())//self.samples_per_track
            while item_drums == item//self.samples_per_track:
                item_drums = np.random.randint(self.__len__())//self.samples_per_track
             
            track_folder_bass = self.foldername_list[item_bass]
            track_folder_vocals = self.foldername_list[item_vocals]
            track_folder_other = self.foldername_list[item_other]
            track_folder_drums = self.foldername_list[item_drums]
        else:
            track_folder_bass = self.foldername_list[item//self.samples_per_track]
            track_folder_vocals =self.foldername_list[item//self.samples_per_track]
            track_folder_other = self.foldername_list[item//self.samples_per_track]
            track_folder_drums = self.foldername_list[item//self.samples_per_track]

#        mixture_filepath = os.path.join(mixture_track_folder,'mixture.wav')
#        target_bass_filepath = os.path.join(track_folder_bass, 'bass.wav')
#        target_vocals_filepath = os.path.join(track_folder_vocals,'vocals.wav')
#        target_other_filepath = os.path.join(track_folder_other,'other.wav')
        # 'other' stem is made up of every source except vocals, bass and drums
        target_other_folders = get_subfoldernames_in_folder(track_folder_other)
        target_other_folders_log_idx = [False if not(folder.endswith('vocals')) and not(folder.endswith('bass')) and not(folder.endswith('drums')) else True for folder in target_other_folders]
        target_other_folders = np.delete(target_other_folders, np.where(target_other_folders_log_idx))
#        target_drums_filepath = os.path.join(track_folder_drums,'drums.wav')
        if self.valid_flag:
            np.random.seed(item)
            temp = self.get_training_sample(track_folder_bass, track_folder_vocals, target_other_folders, track_folder_drums, self.random_mixing, self.augmentations, self.duration)
            np.random.seed()
        else:
            temp = self.get_training_sample(track_folder_bass, track_folder_vocals, target_other_folders, track_folder_drums, self.random_mixing, self.augmentations, self.duration)
        
        return temp


class MSSMUSMoisDBDataset(data.Dataset):
    def __init__(self, 
                 moisdb_foldername_list,
                 moisdb_samples_per_track,
                 enforce_full_mix_percentage,
                 musdb_foldername_list,
                 musdb_samples_per_track,
                 valid_flag,
                 target_str, 
                 random_mix_flag, 
                 augmentation_flag, 
                 duration,
                 rand_seed=None) -> None:
        super().__init__()
        #remove folders that do not contain the target source
        temp_list = np.array(moisdb_foldername_list)
        for folder in moisdb_foldername_list:
            subfolders = []
            subfolders += sorted(glob(os.path.join(folder, '**')))
#            subfolders = get_subfoldernames_in_folder(folder)

            contains_target = [True if target_str in subfolder else False for subfolder in subfolders]
            if not any(contains_target):
                temp_list = np.delete(temp_list,temp_list==folder)
        moisdb_foldername_list = temp_list
        #check if other sources are present in folders
        for folder in moisdb_foldername_list:
            subfolders = []
            subfolders += sorted(glob(os.path.join(folder, '**')))
            contains_bass = [True if 'bass' in subfolder else False for subfolder in subfolders]
            contains_drums = [True if 'drums' in subfolder else False for subfolder in subfolders]
            if not any(contains_bass) or not any(contains_drums):
                temp_list = np.delete(temp_list,temp_list==folder)
        moisdb_foldername_list = temp_list

        self.moisdb_foldername_list = moisdb_foldername_list
        self.musdb_foldername_list = musdb_foldername_list
        self.musdb_samples_per_track = musdb_samples_per_track
        self.musdb_valid_flag = valid_flag
        self.target_str = target_str
        self.random_mixing = random_mix_flag
        self.augmentations = augmentation_flag
        self.duration = duration
        self.length = len(moisdb_foldername_list) + len(musdb_foldername_list)
        self.musdb_len = len(musdb_foldername_list)
        self.moisdb_len = len(moisdb_foldername_list)
        self.moises_samples_per_track = moisdb_samples_per_track
        self.full_mix_percentage = enforce_full_mix_percentage

        rand_seed_musdb = rand_seed
        rand_seed_moises = 2*rand_seed #use different seed for moisesdb, doesn't really matter


        self.musdb_dataset = MSSMUSDBDataset(musdb_foldername_list, 
                                             target_str, 
                                             random_mix_flag,
                                             augmentation_flag,
                                             enforce_full_mix_percentage,
                                             duration, 
                                             musdb_samples_per_track, 
                                             valid_flag,
                                             rand_seed_musdb)


        self.moisdb_dataset = MSSMoisesDBDataset(moisdb_foldername_list, 
                                                 target_str, 
                                                 random_mix_flag, 
                                                 augmentation_flag, 
                                                 duration, 
                                                 moisdb_samples_per_track,
                                                 valid_flag,
                                                 enforce_full_mix_percentage,
                                                 rand_seed_moises)
        
        self.musdb_n_samples = self.musdb_dataset.__len__()
        self.moisdb_n_samples = self.moisdb_dataset.__len__()
        
        assert self.musdb_dataset.fs == self.moisdb_dataset.fs, 'Sampling rate of MUSDB and MoisesDB must be the same'
        self.fs = self.musdb_dataset.fs

    def update_samples_per_track(self, musdb_samples_per_track_new):
        self.musdb_dataset.samples_per_track = musdb_samples_per_track_new
        if musdb_samples_per_track_new != 1:
            # ensure that there are equal amount of samples from musdb and moisesdb
            self.moisdb_dataset.samples_per_track = int(np.ceil(self.musdb_len*musdb_samples_per_track_new/self.moisdb_len))
        else:
            self.moisdb_dataset.samples_per_track = 1
            
        self.musdb_n_samples = self.musdb_dataset.__len__()
        self.moisdb_n_samples = self.moisdb_dataset.__len__()

    def __len__(self):
        return self.musdb_dataset.__len__() + self.moisdb_dataset.__len__()
    
    def __getitem__(self, item):
#        print(item)
#        torch.manual_seed(13)
#        np.random.seed(13)

        if item < self.musdb_n_samples:
#            print("sample #"+str(item)+" from MusDB")
            return self.musdb_dataset.__getitem__(item)
        else:
#            print("sample #"+str(item)+" from MoisesDB")
            return self.moisdb_dataset.__getitem__(item-self.musdb_n_samples)

