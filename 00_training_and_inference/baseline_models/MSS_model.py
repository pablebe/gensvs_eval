import time
from math import ceil
import warnings

import torch
import pytorch_lightning as pl
import torch.distributed as dist
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
from sgmsvs.sgmse import sampling
from sgmsvs.sgmse.sdes import SDERegistry
from sgmsvs.sgmse.backbones import BackboneRegistry
from sgmsvs.sgmse.util.inference import evaluate_model
from sgmsvs.sgmse.util.other import pad_spec, si_sdr

class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for musical source separation enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="score_matching", help="The type of loss function to use.")
        parser.add_argument("--loss_weighting", type=str, default="sigma^2", help="The weighting of the loss function.")
        parser.add_argument("--network_scaling", type=str, default=None, help="The type of loss scaling to use.")
        parser.add_argument("--c_in", type=str, default="1", help="The input scaling for x.")
        parser.add_argument("--c_out", type=str, default="1", help="The output scaling.")
        parser.add_argument("--c_skip", type=str, default="0", help="The skip connection scaling.")
        parser.add_argument("--sigma_data", type=float, default=0.1, help="The data standard deviation.")
        parser.add_argument("--l1_weight", type=float, default=0.001, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--valid_sep_dir", type=str, default=None, help="The directory in which separated validation examples are stored.")
        parser.add_argument("--audio_log_files", nargs='+', type=int, default=None, help="List of audio ids of files to log during training.")
        parser.add_argument("--target_is_accompaniment", action='store_true', default=False, help="Use the accompaniment as target data to diffuse into.")
#        parser.add_argument("--pesq_weight", type=float, default=0.0, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--sr", type=int, default=48000, help="The sample rate of the audio files.")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=0.03, num_eval_files=20, loss_type='score_matching', 
        loss_weighting='sigma^2', network_scaling=None, c_in='1', c_out='1', c_skip='0', sigma_data=0.1, 
        l1_weight=0.001, valid_sep_dir=None, audio_log_files=None, sr=48000, data_module_cls=None, target_is_accompaniment=False, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        kwargs['sr']=sr
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.l1_weight = l1_weight
        self.nolog = kwargs.get('nolog', False)
        self.audio_log_interv = kwargs.get('audio_log_interval', 1)
#        self.pesq_weight = pesq_weight
#        sdr = SignalDistortionRatio()
#        si_sdr = ScaleInvariantSignalDistortionRatio()
#        self.sdr = sdr.forward
#        self.si_sdr = si_sdr.forward
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
        self.network_scaling = network_scaling
        self.c_in = c_in
        self.c_out = c_out
        self.c_skip = c_skip
        self.sigma_data = sigma_data
        self.num_eval_files = num_eval_files
        self.valid_sep_dir = valid_sep_dir
        self.valid_audio_log_files = audio_log_files
        self.sr = sr
        self.valid_ct = 0
        self.accomp_target = target_is_accompaniment
        # Initialize PESQ loss if pesq_weight > 0.0
#        if pesq_weight > 0.0:
#            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
#            for param in self.pesq_loss.parameters():
#                param.requires_grad = False
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.ckpt = None
#        self.temp_ct = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)

        if not(self.nolog):
            #set completed epoch for logging
            try:
                self.trainer.fit_loop.epoch_progress.current.completed=checkpoint['epoch']
            except RuntimeError as e:
                print('No Trainer found, current epoch was not set.')
#        self.ckpt=checkpoint
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
#            self.dnn.load_state_dict(checkpoint['state_dict'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")
        # from thop import profile
        # dummy_batch = self.data_module.train_set.__getitem__(0)
        # x_dummy = dummy_batch[0].to(device=self.device).unsqueeze(1)
        # y_dummy = dummy_batch[1].to(device=self.device).unsqueeze(1)
        # dummy_input =  torch.cat([x_dummy, y_dummy], dim=1).to(device=self.device)
        # dummy_input_2 = torch.randn(dummy_batch[0].shape[0]).to(device=self.device)
        # macs, params = profile(self.dnn, inputs=(dummy_input, dummy_input_2))
        # print("THOP model summary for: masked-based "+str(self.backbone))
        # print(f"MACs/s: {macs / 1e9:.2f} B")
        # print(f"Parameters: {params / 1e6:.2f} M")

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

    def _loss(self, forward_out, x_t, z, t, mean, x):
        """
        Different loss functions can be used to train the score model, see the paper: 
        
        Julius Richter, Danilo de Oliveira, and Timo Gerkmann
        "Investigating Training Objectives for Generative Speech Enhancement"
        https://arxiv.org/abs/2409.10753

        """

        sigma = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "score_matching":
            score = forward_out
            if self.loss_weighting == "sigma^2":
                losses = torch.square(torch.abs(score * sigma + z)) # Eq. (7)
            else:
                raise ValueError("Invalid loss weighting for loss_type=score_matching: {}".format(self.loss_weighting))
            # Sum over spatial dimensions and channels and mean over batch
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        elif self.loss_type == "denoiser":
            score = forward_out
            D = score * sigma.pow(2) + x_t # equivalent to Eq. (10)
            losses = torch.square(torch.abs(D - mean)) # Eq. (8)
            if self.loss_weighting == "1":
                losses = losses
            elif self.loss_weighting == "sigma^2":
                losses = losses * sigma**2
            elif self.loss_weighting == "edm":
                losses = ((sigma**2 + self.sigma_data**2)/((sigma*self.sigma_data)**2))[:, None, None, None] * losses
            else:
                raise ValueError("Invalid loss weighting for loss_type=denoiser: {}".format(self.loss_weighting))
            # Sum over spatial dimensions and channels and mean over batch
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))     
        elif self.loss_type == "data_prediction":
            x_hat = forward_out
            B, C, F, T = x.shape

            # losses in the time-frequency domain (tf)
            losses_tf = (1/(F*T))*torch.square(torch.abs(x_hat - x))
            losses_tf = torch.mean(0.5*torch.sum(losses_tf.reshape(losses_tf.shape[0], -1), dim=-1))

            # losses in the time domain (td)
            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
            x_td = self.to_audio(x.squeeze(), target_len)
            losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
            losses_l1 = torch.mean(0.5*torch.sum(losses_l1.reshape(losses_l1.shape[0], -1), dim=-1))

            loss = losses_tf + self.l1_weight * losses_l1
        else:
            raise ValueError("Invalid loss type: {}".format(self.loss_type))

        return loss

    def _step(self, batch, batch_idx):
        x, y, audio_x, audio_y = batch

        if self.accomp_target:
            x = y-x # make accompaniment target

        #reshape => fuse channel and batch dimensions and unsqueeze so dimension fits for sde.marginal_prob()
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3]).unsqueeze(1)
        y = y.reshape(y.shape[0]*y.shape[1], y.shape[2], y.shape[3]).unsqueeze(1)
        y = pad_spec(y, mode="reflection")
        x = pad_spec(x, mode="reflection")

        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, y, t)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z
        forward_out = self(x_t, y, t)
        
        loss = self._loss(forward_out, x_t, z, t, mean, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if batch_idx == 0 and self.num_eval_files != 0:
            # Split the evaluation files among the GPUs
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

                x_hat = self.enhance(y, N=self.sde.N)
                x_hat = torch.from_numpy(x_hat)

                if self.accomp_target:
                    x_hat = y-x_hat


                if self.valid_sep_dir is not None:
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
                    temp_test_sisdr = 0
                    temp_multi_res = 0
                    for ii in range(x.shape[0]):
                        temp_sdr += self.sdr(x_hat[ii,:], x[ii,:])        
                        temp_sisdr += self.si_sdr(x_hat[ii,:], x[ii,:])
                        temp_test_sisdr += si_sdr(x[ii,:], x_hat[ii,:])
                        temp_multi_res += self.multi_res_loss(x_hat[ii,:].unsqueeze(0).unsqueeze(0), x[ii,:].unsqueeze(0).unsqueeze(0))
                    sdr_sum += temp_sdr.item()/x.shape[0]
                    si_sdr_sum += temp_sisdr.item()/x.shape[0]
                    test_sisdr += temp_test_sisdr.item()/x.shape[0]
                    multi_res_loss_sum += temp_multi_res.item()/x.shape[0]
                else:
                    sdr_sum += self.sdr(torch.from_numpy(x_hat), torch.from_numpy(x))
                    si_sdr_sum += self.si_sdr(x_hat,x)
                    test_sisdr += si_sdr(x, x_hat)
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

        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    


    def forward(self, x_t, y, t):
        """
        The model forward pass. In [1] and [2], the model estimates the score function. In [3], the model estimates 
        either the score function or the target data for the Schrödinger bridge (loss_type='data_prediction').
        
        [1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, and  Timo Gerkmann 
            "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models"
            IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023. 

        [2] Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker, Bunlong Lay, Shinji Watanabe, Alexander Richard, and Timo Gerkmann
            "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation"
            ISCA Interspecch, Kos, Greece, Sept. 2024. 

        [3] Julius Richter, Danilo de Oliveira, and Timo Gerkmann
            "Investigating Training Objectives for Generative Speech Enhancement"
            https://arxiv.org/abs/2409.10753

        """

        # In [3], we use new code with backbone='ncsnpp_v2':
        if self.backbone == "ncsnpp_v2":
            F = self.dnn(self._c_in(t) * x_t, self._c_in(t) * y, t)
            
            # Scaling the network output, see below Eq. (7) in the paper
            if self.network_scaling == "1/sigma":
                std = self.sde._std(t)
                F = F / std[:, None, None, None]
            elif self.network_scaling == "1/t":
                F = F / t[:, None, None, None]

            # The loss type determines the output of the model
            if self.loss_type == "score_matching":
                score = self._c_skip(t) * x_t + self._c_out(t) * F
                return score
            elif self.loss_type == "denoiser":
                sigmas = self.sde._std(t)[:, None, None, None]
                score = (F - x_t) / sigmas.pow(2)
                return score
            elif self.loss_type == 'data_prediction':
                x_hat = self._c_skip(t) * x_t + self._c_out(t) * F
                return x_hat

        # In [1] and [2], we use the old code:
        else:
            dnn_input = torch.cat([x_t, y], dim=1)            
            score = -self.dnn(dnn_input, t)
            return score

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        elif self.c_in == "edm":
            sigma = self.sde._std(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_in type: {}".format(self.c_in))
    
    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        elif self.c_out == "sigma":
            return self.sde._std(t)[:, None, None, None]
        elif self.c_out == "1/sigma":
            return 1.0 / self.sde._std(t)[:, None, None, None] 
        elif self.c_out == "edm":
            sigma = self.sde._std(t)
            return ((sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_out type: {}".format(self.c_out))
    
    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        elif self.c_skip == "edm":
            sigma = self.sde._std(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_skip type: {}".format(self.c_skip))

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N

        return sampling.get_sb_sampler(sde, self, y=y, sampler_type=sampler_type, **kwargs)

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

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=2, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max()#.item()
        y = y / norm_factor
        if y.shape[0]>1:
            Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 1)
        else:
            Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
  
        Y = pad_spec(Y, mode="reflection")

        # SGMSE sampling with OUVE SDE
        if self.sde.__class__.__name__ == 'OUVESDE':
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), N=N, 
                    corrector_steps=corrector_steps, snr=snr, intermediate=False, **kwargs)
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
            else:
                raise ValueError("Invalid sampler type for SGMSE sampling: {}".format(sampler_type))
            
        # Schrödinger bridge sampling with VE SDE
        elif self.sde.__class__.__name__ == 'SBVESDE':
            sampler = self.get_sb_sampler(sde=self.sde, y=Y.cuda(), sampler_type=self.sde.sampler_type)
        else:
            raise ValueError("Invalid SDE type for speech enhancement: {}".format(self.sde.__class__.__name__))

        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/self.sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
