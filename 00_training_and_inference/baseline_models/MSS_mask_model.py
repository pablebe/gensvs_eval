##-------------------------------------------------------------------------------------------------
# This file contains the MaskingModel class, which is used for training 
# the discriminative mask-based models for singing voice separation.
# It was tested for HTDemucs and the Mel-RoFormer models.
# The whole framework including this class was adapted from 
# the work of Richter etal. in [1]. The sgmse framework of [1] published 
# in [2] was used to set up the whole training and inference pipeline.
#
# References:
# [1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann. 
# "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models", 
# IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023.
# [2] Julius Richter et al., 
# "Speech Enhancement and Dereverberation with Diffusion-based Generative Models",
# [Online], Github, URL: https://github.com/julius-richter/sgmse, Accessed: 2025-05-09,

import time
from math import ceil
import warnings
import argparse
import torch
import pytorch_lightning as pl
import torch.distributed as dist
import torch.nn.functional as F

import numpy as np
import tqdm
import soundfile
import os
import wandb

from torchaudio import load
from torch_ema import ExponentialMovingAverage
from librosa import resample
from torchmetrics.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio, ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from auraloss.freq import MultiResolutionSTFTLoss
from einops import rearrange
from baseline_models.backbones import BackboneRegistry



def str2bool(value):
    """Convert string 'True'/'False' to boolean True/False."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class MaskingModel(pl.LightningModule):
    
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--stereo", type=str2bool, default=False, help="Whether the model is trained on stereo data (False by default)")
        parser.add_argument("--bypass_lr_scheduler", type=str2bool, default=False, help="Whether to bypass the learning rate scheduler (False by default)")
        parser.add_argument("--lr_scheduler_patience", type=int, default=2, help="The patience for the learning rate scheduler (2 by default)")
        parser.add_argument("--lr_scheduler_reduce_factor", type=float, default=0.95, help="The factor to reduce the learning rate by (0.95 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for musical source separation enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--valid_sep_dir", type=str, default=None, help="The directory in which separated validation examples are stored.")
        parser.add_argument("--audio_log_files", nargs='+', type=int, default=None, help="List of audio ids of files to log during training.")
        parser.add_argument("--sr", type=int, default=48000, help="The sample rate of the audio files.")
        parser.add_argument("--stft_n_fft", type=int, default=2048, help="n_fft for the stft.")
        parser.add_argument("--loss_type", type=str, default='combined_multi_resolution_loss', help="The type of loss to use. Default is 'combined_multi_resolution_loss'.")
        parser.add_argument("--multi_stft_resolution_loss_weight", type=float, default=1.0, help="weight for the multi stft resolution loss.")
        parser.add_argument("--multi_stft_resolutions_window_sizes", nargs='+', type=int, default=[4096, 2048, 1024, 512, 256], help="List of audio ids of files to log during training.")
        parser.add_argument("--multi_stft_hop_size", type=int, default=147, help="hop size for the multi stft.")
        parser.add_argument("--multi_stft_normalized", type=str2bool, default=False, help="whether to normalize the multi stft.")
        parser.add_argument("--accumulated_grad_batches", type=int, default=1, help="Number of batches to accumulate gradients over.")
        parser.add_argument("--masked_mse_coarse", type=str2bool, default=True, help="Whether to use masked mse loss.")
        parser.add_argument("--masked_mse_q", type=float, default=0.95, help="Quantile for masked mse loss.")
        return parser

    def __init__(
        self, backbone, lr=1e-4, lr_scheduler_patience=2, lr_scheduler_reduce_factor=0.95, bypass_lr_scheduler=False, ema_decay=0.999, num_eval_files=20, loss_type='multi_resolution_loss', 
        valid_sep_dir=None, audio_log_files=None, sr=48000, data_module_cls=None, stft_n_fft=2048,
        multi_stft_resolution_loss_weight=1.0, multi_stft_resolutions_window_sizes=[4096, 2048, 1024, 512, 256],
        multi_stft_hop_size=147, multi_stft_normalized=False, 
        masked_mse_coarse=True, masked_mse_q=0.95, **kwargs
    ):
        """
        Create a new MaskingModel.

        Args:
            backbone: Backbone DNN that which estimates mask for separation.
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        kwargs['sr']=sr
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        kwargs.update({'stft_n_fft':stft_n_fft, 'sample_rate':sr})
        self.dnn = dnn_cls(**kwargs)
        self.multi_stft_n_fft = stft_n_fft
        self.window_str = kwargs.get('window', 'hann')
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )
        self.masked_mse_coarse = masked_mse_coarse
        self.masked_mse_q = masked_mse_q
        
        self.lr = lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_reduce_factor = lr_scheduler_reduce_factor
        self.bypass_lr_scheduler = bypass_lr_scheduler
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.loss_type = loss_type
        self.nolog = kwargs.get('nolog', False)
        self.audio_log_interv = kwargs.get('audio_log_interval', 1)

        self.sdr = signal_distortion_ratio
        self.si_sdr = scale_invariant_signal_distortion_ratio
        multi_res_loss = MultiResolutionSTFTLoss(
                                                    fft_sizes=[1024, 2048, 8192],
                                                    hop_sizes=[256, 512, 2048],
                                                    win_lengths=[1024, 2048, 8192],
                                                    scale="mel",
                                                    n_bins=128,
                                                    sample_rate=sr,
                                                    perceptual_weighting=True,
                                                )
        self.multi_res_loss = multi_res_loss.forward

        self.num_eval_files = num_eval_files
        self.valid_sep_dir = valid_sep_dir
        self.valid_audio_log_files = audio_log_files
        self.sr = sr
        self.valid_ct = 0
 
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.ckpt = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.bypass_lr_scheduler:

            opt_dict = {
                        'optimizer': optimizer
                        }
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.lr_scheduler_patience,
                                  factor=self.lr_scheduler_reduce_factor)

            opt_dict = {
                        'optimizer': optimizer, 
                        'lr_scheduler': {'scheduler': scheduler, 'monitor': 'sdr', 'interval': 'epoch', 'frequency': 1, 'strict': True}
                        }
        return opt_dict

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())


    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)

        if not(self.nolog):
            #set completed epoch for logging
            try:
                self.trainer.fit_loop.epoch_progress.current.completed=checkpoint['epoch']
            except RuntimeError as e:
                print('No Trainer found, current epoch was not set.')
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.dnn.parameters())        # store current params in EMA
                self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, target, recon_audio):
        """
        Different loss functions can be used to train the masking model 
        Args:
            target: The target vocal signal
            recon_audio: The separated vocal signal
        Returns:
            loss: The loss value
        """
        device = target.device

        if self.loss_type == "combined_multi_resolution_loss":
            if self.dnn.num_stems > 1:
                assert target.ndim == 4 and target.shape[1] == self.dnn.num_stems

            if target.ndim == 2:
                target = rearrange(target, '... t -> ... 1 t')

            target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

            loss = F.l1_loss(recon_audio, target)

            multi_stft_resolution_loss = 0.

            for window_size in self.multi_stft_resolutions_window_sizes:

                if self.window_str == 'hann':
                    win = torch.windows.hann(window_size, device=device)
                elif self.window_str == 'sqrt_hann':
                    win = torch.sqrt(torch.windows.hann(window_size, device=device))
                else:
                    raise ValueError("Invalid window type: {}".format(self.window_str))

                res_stft_kwargs = dict(
                    n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                    win_length=window_size,
                    return_complex=True,
                    window=win,
                    **self.multi_stft_kwargs,
                )

                recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
                target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

                multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

            weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

            total_loss = loss + weighted_multi_resolution_loss
            return total_loss, loss, weighted_multi_resolution_loss
        elif self.loss_type == "multi_resolution_loss":

            if self.dnn.num_stems > 1:
                assert target.ndim == 4 and target.shape[1] == self.dnn.num_stems

            if target.ndim == 2:
                target = rearrange(target, '... t -> ... 1 t')

            target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

            multi_stft_resolution_loss = 0.
            

            for window_size in self.multi_stft_resolutions_window_sizes:

                if self.window_str == 'hann':
                    win = torch.windows.hann(window_size, device=device)
                elif self.window_str == 'sqrt_hann':
                    win = torch.sqrt(torch.windows.hann(window_size, device=device))
                else:
                    raise ValueError("Invalid window type: {}".format(self.window_str))

                res_stft_kwargs = dict(
                    n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                    win_length=window_size,
                    return_complex=True,
                    window=win,
                    **self.multi_stft_kwargs,
                )

                recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
                target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

                multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

            return multi_stft_resolution_loss
        elif self.loss_type == "masked_mse":
                
                loss = torch.nn.MSELoss(reduction='none')(recon_audio, target.unsqueeze(1)).transpose(0, 1)
                batch_size = recon_audio.shape[0]
                if self.masked_mse_coarse:
                    loss = torch.mean(loss, dim=(-1, -2))
                    
                loss = loss.reshape(loss.shape[0], -1)
                L = loss.detach()
                quantile = torch.quantile(L, self.masked_mse_q, interpolation='linear', dim=1, keepdim=True)
                mask = L < quantile
                if batch_size>1:
                    return (loss * mask).mean()
                else:
                    return loss.mean()
                
        elif self.loss_type == "l1_loss":
                loss = torch.functional.F.l1_loss(recon_audio, target.unsqueeze(1))
                return loss
        else:
            raise ValueError("Invalid loss type: {}".format(self.loss_type))


    def _step(self, batch, batch_idx):
        _, _, audio_x, audio_y = batch

        #reshape => fuse channel and batch dimensions and unsqueeze so dimension fits for sde.marginal_prob()
        enh = self.dnn(audio_y)
        loss = self._loss(audio_x, enh)
        return loss

    def training_step(self, batch, batch_idx):
        if self.loss_type == 'combined_multi_resolution_loss':
            total_loss, l1_loss, multi_resolution_loss = self._step(batch, batch_idx)
        else:
            total_loss  = self._step(batch, batch_idx)
        self.log('train_loss', total_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if batch_idx == 0 and self.num_eval_files != 0:
            eval_files_per_gpu = self.num_eval_files // world_size
            # Select the files for this GPU     
            if rank == world_size - 1:
                first_valid_file = rank*eval_files_per_gpu
                last_valid_file = self.num_eval_files
            else:   
                first_valid_file = rank*eval_files_per_gpu
                last_valid_file = (rank+1)*eval_files_per_gpu
            np.random.seed(2*self.data_module.rand_seed)
            rand_idx = np.arange(self.data_module.valid_set.__len__())
            np.random.shuffle(rand_idx)
            rand_idx = rand_idx[:self.num_eval_files]
            np.random.seed()
            # Evaluate the performance of the model
            sdr_sum = 0; si_sdr_sum = 0; multi_res_loss_sum = 0; test_sisdr = 0
            for item_id in tqdm.tqdm(rand_idx[first_valid_file:last_valid_file],desc='Validation on GPU '+str(rank)):
                # Load the clean and noisy speech

                y, x, target_rms = self.data_module.valid_set.dataset[item_id]

                x = x.squeeze().numpy()
                y = y.squeeze().numpy()


                if self.data_module.valid_set.dataset.fs != self.sr:
                    #resample audio
                    y = resample(y, orig_sr=self.data_module.valid_set.dataset.fs, target_sr=self.sr)
                    x = resample(x, orig_sr=self.data_module.valid_set.dataset.fs, target_sr=self.sr)
                y = torch.from_numpy(y)
                x = torch.from_numpy(x)

                if not self.data_module.train_mono:
                    y = y.unsqueeze(0)
                    x = x.unsqueeze(0)

                y = y.to(self.device)
                x = x.to(self.device)
                x_hat = self.enhance(y)

                y = y.squeeze().cpu()
                x = x.squeeze().cpu()
                x_hat = x_hat.squeeze().cpu()

                if self.valid_sep_dir is not None:
                    #if validation directory is set, save the mixture, target and separated files
                    os.makedirs(os.path.join(self.valid_sep_dir,'target'), exist_ok=True)
                    os.makedirs(os.path.join(self.valid_sep_dir,'mixture'), exist_ok=True)
                    os.makedirs(os.path.join(self.valid_sep_dir,'separated'), exist_ok=True)

                    soundfile.write(os.path.join(self.valid_sep_dir,'mixture','mixture_fileid_'+str(item_id.item())+'.wav'), y.T, self.sr)
                    soundfile.write(os.path.join(self.valid_sep_dir,'target','target_fileid_'+str(item_id.item())+'.wav'), x.T, self.sr)
                    soundfile.write(os.path.join(self.valid_sep_dir,'separated','separated_fileid_'+str(item_id.item())+'.wav'), x_hat.T, self.sr)

                if not(self.nolog) and (self.valid_audio_log_files is not None) and (item_id in self.valid_audio_log_files) and (self.current_epoch % self.audio_log_interv)==0:
                    log_data_dict = {}
                    if self.valid_ct==0:
                    # only log mixture and target for the first time
                        log_data_dict["file #"+str(item_id.item())] = [wandb.Audio(y.T, self.sr, caption='mixture'), wandb.Audio(x.T,self.sr,caption='target'), wandb.Audio(x_hat.T,self.sr, caption='separated')]                       
                    else:
                        log_data_dict["file #"+str(item_id.item())] = [wandb.Audio(x_hat.T,self.sr, caption='separated')]
                    
                    wandb.log(log_data_dict)

                if x.shape[0] > 1:
                    temp_sdr = 0
                    temp_sisdr = 0
                    temp_multi_res = 0
                    for ii in range(x.shape[0]):
                        temp_sdr += self.sdr(x_hat[ii,:], x[ii,:])        
                        temp_sisdr += self.si_sdr(x_hat[ii,:], x[ii,:])
                        temp_multi_res += self.multi_res_loss(x_hat[ii,:].unsqueeze(0).unsqueeze(0), x[ii,:].unsqueeze(0).unsqueeze(0))
                    sdr_sum += temp_sdr.item()/x.shape[0]
                    si_sdr_sum += temp_sisdr.item()/x.shape[0]
                    multi_res_loss_sum += temp_multi_res.item()/x.shape[0]
                else:
                    sdr_sum += self.sdr(torch.from_numpy(x_hat), torch.from_numpy(x))
                    si_sdr_sum += self.si_sdr(x_hat,x)
                    multi_res_loss_sum += self.multi_res_loss(x_hat.unsqueeze(0), x.unsqueeze(0))
            sdr_avg = sdr_sum / len(rand_idx[first_valid_file:last_valid_file])
            si_sdr_avg = si_sdr_sum / len(rand_idx[first_valid_file:last_valid_file])
            multi_res_loss_avg = multi_res_loss_sum / len(rand_idx[first_valid_file:last_valid_file])
            self.log('sdr', sdr_avg, on_step=False, on_epoch=True, sync_dist=True)
            self.log('si_sdr', si_sdr_avg, on_step=False, on_epoch=True, sync_dist=True)
            self.log('multi_res_loss', multi_res_loss_avg, on_step=False, on_epoch=True, sync_dist=True)
        else:
            sdr_avg = None
            si_sdr_avg = None
            multi_res_loss_avg = None

        if self.loss_type == 'combined_multi_resolution_loss':
            total_loss, l1_loss, multi_resolution_loss = self._step(batch, batch_idx)
            self.log('valid_l1-loss', l1_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log('valid_multi_resolution_loss', multi_resolution_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log('combined_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)

        elif self.loss_type == 'masked_mse':
            total_loss  = self._step(batch, batch_idx)
            self.log('mse_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)
            
        elif self.loss_type == 'l1_loss':
            total_loss = self._step(batch, batch_idx)
            self.log('l1_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss
    
    def forward(self, y):
        """
        Forward pass through the model. This is called by the trainer during training and validation.
        Args:
            y: The input musical mixture
        Returns:
            x_hat: The separated vocal signal
        """
        x_hat = self.dnn(y)
        return x_hat


    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, timeit=False, **kwargs):
        """
        One-call singing voice separation.
        """
        start = time.time()
        x_hat = self.dnn(y)
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/self.sr)
            return x_hat, rtf
        else:
            return x_hat
