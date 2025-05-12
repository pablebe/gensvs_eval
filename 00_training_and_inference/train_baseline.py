# Copyright (c) 2025
#   Licensed under the MIT license.

# Adapted from https://github.com/sp-uhh/sgmse under the MIT license.


import os
#os.environ["WANDB_API_KEY"] = "" # set your wandb key here
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set your CUDA device here
os.environ["WANDB__SERVICE_WAIT"] = "300"

import torch
import wandb
import argparse
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join

# Set CUDA architecture list and float32 matmul precision high
from sgmsvs.sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

from baseline_models.backbones.shared import BackboneRegistry
from sgmsvs.data_module import SpecsDataModule
#from sgmse.sdes import SDERegistry
from baseline_models.MSS_mask_model import MaskingModel

#wandb.login() #!!!Uncomment this line if you want to login to wandb!!!

def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups

if __name__ == '__main__':
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="mel_band_roformer")
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging.")
          parser_.add_argument("--audio_log_interval", type=int, default=5, help="Log audio every n epochs.")
          parser_.add_argument("--wandb_name", type=str, default=None, help="Name for wandb logger. If not set, a random name is generated.")
          parser_.add_argument("--ckpt", type=str, default=None, help="Resume training from checkpoint.")
          parser_.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
          parser_.add_argument("--save_ckpt_interval", type=int, default=50000, help="Save checkpoint interval.")
          parser_.add_argument("--start_with_validation", action='store_true', help="Start with validation before training.")
          parser_.add_argument("--run_id", type=str, default=None, help="Set run id so distributed training is logged on same run")
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
     trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
     trainer_parser.add_argument("--devices", default="auto", help="How many gpus to use.")
     trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
     trainer_parser.add_argument("--max_epochs", type=int, default=-1, help="Number of epochs to train.")

     MaskingModel.add_argparse_args(
          parser.add_argument_group("MaskingModel", description=MaskingModel.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     #set stft params from model to data-module such that they are consistent
     args.stft_n_fft = args.n_fft
     args.stft_win_length = args.n_fft
     args.stft_hop_length = args.hop_length
     args.stft_window_fn = args.window
     args.sources = [args.target_str]
     
     if args.train_mono:
          args.num_channels = 1
          args.stereo = False
     else:
          args.stereo = True


     arg_groups = get_argparse_groups(parser)

     kwargs = {'nolog': args.nolog, 'audio_log_interval': args.audio_log_interval, 'sr': args.sr}
     
     model = MaskingModel(
          backbone=args.backbone, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['MaskingModel']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule']),
               **kwargs
          }
     )
     if torch.cuda.is_available() and args.devices == "auto":
          model = model.to("cuda")

     if args.nolog:
          logger = None
     else:
          logger = WandbLogger(project="sgmse-MSS", log_model=False, save_dir="logs", name=args.wandb_name, id=args.run_id)
          logger.experiment.log_code(".")
          wandb.init(project="sgmse-MSS", name=args.wandb_name, id=args.run_id, dir="logs")

     # Set up callbacks for logger
     if logger != None:
          callbacks = [ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)), save_last=True, 
               filename='{epoch}-last')]
          callbacks += [ModelCheckpoint(dirpath=join(args.log_dir, f'{str(logger.version)}-{args.wandb_name}'),
               filename='{step}', save_top_k=-1, every_n_train_steps=args.save_ckpt_interval)]
          if args.num_eval_files:
               checkpoint_callback_sdr = ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)), 
                    save_top_k=1, monitor="sdr", mode="max", filename='{epoch}-{sdr:.2f}')
               checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)), 
                    save_top_k=1, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
               checkpoint_callback_multi_res_loss = ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)),
                    save_top_k=1, monitor="multi_res_loss", mode="min", filename='{epoch}-{multi_res_loss:.2f}')
               callbacks += [checkpoint_callback_sdr, checkpoint_callback_si_sdr, checkpoint_callback_multi_res_loss]
     else:
          callbacks = None

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer(
          **vars(arg_groups['Trainer']),
          strategy="ddp", logger=logger,
          log_every_n_steps=10, num_sanity_val_steps=0,
          callbacks=callbacks
     )

     if args.start_with_validation:
          model.eval()
          trainer.validate(model, ckpt_path=args.ckpt)
          model.valid_ct += 1
     
  
     # get model complexity info
     from ptflops import get_model_complexity_info

     macs_ptflops, params_ptflops = get_model_complexity_info(
          model.dnn,
          (2,1*args.sr),  # Input size without batch dim
          as_strings=False,
          print_per_layer_stat=True,
          verbose=True
     )
     
     print("PTFLOPS model summary for: masked-based "+str(args.backbone))
     print(f"MACs/s: {macs_ptflops / 1e9:.2f} B")
     print(f"Parameters: {params_ptflops / 1e6:.2f} M")
     
     # Train the model
     trainer.fit(model, ckpt_path=args.ckpt)

