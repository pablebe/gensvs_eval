import torch
import numpy as np
import soundfile
import tqdm
import shutil
import os
import matlab.engine
import pandas as pd
import subprocess
import fast_bss_eval
from glob import glob
from torchmetrics.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from auraloss.freq import MultiResolutionSTFTLoss
from argparse import ArgumentParser
from librosa import resample


from audiobox_aesthetics.infer import initialize_predictor
from xls_r_sqa.config import XLSR_2B_TRANSFORMER_32DEEP_CONFIG
from xls_r_sqa.e2e_model import E2EModel


eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath('./01_evaluation_and_correlation/peass_v2.0.1'))
eng.compile


audiobox_predictor = initialize_predictor()

#use these flags to select which metrics to calculate
CALCULATE_PEASS_BSS_EVAL_AND_MRES = True 
CALCULATE_SINGMOS_XLS_R = False
CALCULATE_VISQOL = False
CALCULATE_AUDIOBOX = False

REG_CONST = 1e-7
PERM_FLAG = False
FILT_LEN = 1024

visqol_cl_api = './bazel-bin/visqol'
    


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--mixture_dir', type=str, required=True, help='Path to mixture audio directory')
    parser.add_argument('--target_dir', type=str, required=True, help='Path to target audio directory')
    parser.add_argument('--separated_dir_sgmsvs_from_scratch', type=str, required=True, help='Path to separated audio from sgmse model')
    parser.add_argument('--separated_dir_melroform_bigvgan', type=str, required=True, help='Path to separated audio from melroformer with bigvganr refinement')
    parser.add_argument('--separated_dir_melroform_small', type=str,  required=True, help='Path to separated audio melroformer model')
    parser.add_argument('--separated_dir_melroform_large', type=str, required=True, help='Path to separated audio from sgmse model')
    parser.add_argument('--separated_dir_htdemucs', type=str, required=True, help='Path to separated audio from hybrid transformer demucs model trained MusDBHQ+800 extra songs')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for results')
    parser.add_argument('--sr', type=int, default=44100, required=True, help='sample rate of audio files')

    args = parser.parse_args()
    output_path = args.output_folder
    visqol_temp_folder = os.path.join(output_path,'tmp', 'visqol_temp')
    MatDestDir = os.path.join(output_path,'tmp', 'peass_temp')
    os.makedirs(MatDestDir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(visqol_temp_folder, exist_ok=True)
    sdr = signal_distortion_ratio
    si_sdr = scale_invariant_signal_distortion_ratio
    
    multi_res_loss = MultiResolutionSTFTLoss(fft_sizes=[256, 512, 1024, 2048, 4096],
                                             win_lengths=[256, 512, 1024, 2048, 4096],
                                             hop_sizes=[64, 128,  256, 512, 1024],
                                             sample_rate=args.sr, 
                                             perceptual_weighting=True)

                                            
    singmos_predictor = torch.hub.load("South-Twilight/SingMOS:v0.2.0", "singing_ssl_mos", trust_repo=True)
    
    xls_model = E2EModel(
                        config=XLSR_2B_TRANSFORMER_32DEEP_CONFIG,
                        xlsr_layers=10,
                        auto_download=True,# <-- default is True
                        dataset_variant="subset",
                    )
    
    xls_model.eval()
    
   
    mixture_files = sorted(glob(os.path.join(args.mixture_dir, '*.wav')))
    target_files = sorted(glob(os.path.join(args.target_dir, '*.wav')))
    sep_files_sgmsvs_from_scratch = sorted(glob(os.path.join(args.separated_dir_sgmsvs_from_scratch, '*.wav')))
    sep_files_melroform_bigvgan = sorted(glob(os.path.join(args.separated_dir_melroform_bigvgan, '*.wav')))
    sep_files_melroform_small = sorted(glob(os.path.join(args.separated_dir_melroform_small, '*.wav')))
    sep_files_melroform_large = sorted(glob(os.path.join(args.separated_dir_melroform_large, '*.wav')))
    sep_files_htdemucs = sorted(glob(os.path.join(args.separated_dir_htdemucs, '*.wav')))

    sdr_scores_noisy = []
    si_sdr_scores_noisy = []
    multi_res_loss_scores_noisy = []
    sir_scores_noisy = []
    sar_scores_noisy = []
    isr_scores_noisy = []
    ops_scores_noisy = []
    tps_scores_noisy = []
    ips_scores_noisy = []
    aps_scores_noisy = []
    singmos_scores_noisy = []
    visqol_scores_noisy = []
    meta_aes_pq_scores_noisy = []
    meta_aes_cu_scores_noisy = []
    xls_r_sqa_scores_noisy = []

    sdr_scores_sgmsvs_scratch = []
    si_sdr_scores_sgmsvs_scratch = []
    multi_res_loss_scores_sgmsvs_scratch = []
    sir_scores_sgmsvs_scratch = []
    sar_scores_sgmsvs_scratch = []
    isr_scores_sgmsvs_scratch = []
    ops_scores_sgmsvs_scratch = []
    tps_scores_sgmsvs_scratch = []
    ips_scores_sgmsvs_scratch = []
    aps_scores_sgmsvs_scratch = []
    singmos_scores_sgmsvs_scratch = []
    visqol_scores_sgmsvs_scratch = []
    meta_aes_pq_scores_sgmsvs_scratch = []
    meta_aes_cu_scores_sgmsvs_scratch = []
    xls_r_sqa_scores_sgmsvs_scratch = []
    
    sdr_scores_melroform_bigvgan = []
    si_sdr_scores_melroform_bigvgan = []
    multi_res_loss_scores_melroform_bigvgan = []
    sir_scores_melroform_bigvgan = []
    sar_scores_melroform_bigvgan = []
    isr_scores_melroform_bigvgan = []
    ops_scores_melroform_bigvgan = []
    tps_scores_melroform_bigvgan = []
    ips_scores_melroform_bigvgan = []
    aps_scores_melroform_bigvgan = []
    singmos_scores_melroform_bigvgan = []
    visqol_scores_melroform_bigvgan = []
    meta_aes_pq_scores_melroform_bigvgan = []
    meta_aes_cu_scores_melroform_bigvgan = []
    xls_r_sqa_scores_melroform_bigvgan = []

    sdr_scores_melroform_small = []
    si_sdr_scores_melroform_small = []
    multi_res_loss_scores_melroform_small = []
    sir_scores_melroform_small = []
    sar_scores_melroform_small = []
    isr_scores_melroform_small = []
    ops_scores_melroform_small = []
    tps_scores_melroform_small = []
    ips_scores_melroform_small = []
    aps_scores_melroform_small = []
    singmos_scores_melroform_small = []
    visqol_scores_melroform_small = []
    meta_aes_pq_scores_melroform_small = []
    meta_aes_cu_scores_melroform_small = []
    xls_r_sqa_scores_melroform_small = []
    
    sdr_scores_melroform_large = []
    si_sdr_scores_melroform_large = []
    multi_res_loss_scores_melroform_large = []
    sir_scores_melroform_large = []
    sar_scores_melroform_large = []
    isr_scores_melroform_large = []
    ops_scores_melroform_large = []
    tps_scores_melroform_large = []
    ips_scores_melroform_large = []
    aps_scores_melroform_large = []
    singmos_scores_melroform_large = []
    visqol_scores_melroform_large = []
    meta_aes_pq_scores_melroform_large = []
    meta_aes_cu_scores_melroform_large = []
    xls_r_sqa_scores_melroform_large = []

    sdr_scores_htdemucs = []
    si_sdr_scores_htdemucs = []
    multi_res_loss_scores_htdemucs = []
    sir_scores_htdemucs = []
    sar_scores_htdemucs = []
    isr_scores_htdemucs = []
    ops_scores_htdemucs = []
    tps_scores_htdemucs = []
    ips_scores_htdemucs = []
    aps_scores_htdemucs = []
    singmos_scores_htdemucs = []
    visqol_scores_htdemucs = []
    meta_aes_pq_scores_htdemucs = []
    meta_aes_cu_scores_htdemucs = []
    xls_r_sqa_scores_htdemucs = []

    file_id_csv = []
    for mixture_path in tqdm.tqdm(mixture_files, desc='Calculating Metrics'):
        file_id = mixture_path.split(os.path.sep)[-1].split(".")[0].split("mixture_")[-1]
        file_id_csv.append(file_id)

    
    pd_file_order = pd.DataFrame(file_id_csv, columns=['file_id'])
    pd_file_order.to_csv(os.path.join(output_path, 'file_id_order.csv'))


    for mixture_path in tqdm.tqdm(mixture_files, desc='Calculating Metrics'):

        file_id = mixture_path.split(os.path.sep)[-1].split('mixture_')[-1].split('.wav')[0]
        
        target_file = [f for f in target_files if file_id in f][0]
        sep_file_sgmsvs_from_scratch = [f for f in sep_files_sgmsvs_from_scratch if file_id in f][0]
        sep_file_melroform_bigvgan = [f for f in sep_files_melroform_bigvgan if file_id in f][0]
        sep_file_melroform_small = [f for f in sep_files_melroform_small if file_id in f][0]
        sep_file_melroform_large = [f for f in sep_files_melroform_large if file_id in f][0]
        sep_file_htdemucs = [f for f in sep_files_htdemucs if file_id in f][0]

        print("----------------------------------------")
        print('mixture:             '+mixture_path.split(os.path.sep)[-1])  
        print('target:              '+target_file.split(os.path.sep)[-1])
        print('melroformer-small:   '+sep_file_melroform_small.split(os.path.sep)[-1])
        print('melroformer-large:   '+sep_file_melroform_large.split(os.path.sep)[-1])
        print('melroformer+bigvgan: '+sep_file_melroform_bigvgan.split(os.path.sep)[-1])
        print('sgmsvs (scratch):    '+sep_file_sgmsvs_from_scratch.split(os.path.sep)[-1])
        print('htdemucs:            '+sep_file_htdemucs.split(os.path.sep)[-1])
        print("----------------------------------------")

        mixture, sr = soundfile.read(mixture_path)
        if len(mixture.shape)<2:
            two_ch_flag = False
            mixture = mixture[:,None]
            CH=1
        else:
            two_ch_flag = True
            CH=2
        mixture = mixture.T
        
        mixture_48k = resample(mixture, orig_sr=sr, target_sr=48e3)
        
        target, sr_target = soundfile.read(target_file)
        if not(two_ch_flag):
            assert len(target.shape) == 1, "Target file is not mono!"
            target = target[:,None]
        
        target = target.T

        target_48k = resample(target, orig_sr=sr_target, target_sr=48e3)

        sep_sgmsvs_scratch, sr_sgmsvs_scratch = soundfile.read(sep_file_sgmsvs_from_scratch)
        if not(two_ch_flag):
            assert len(sep_sgmsvs_scratch.shape) == 1, "Separated file is not mono!"
            sep_sgmsvs_scratch = sep_sgmsvs_scratch[:,None]    
        sep_sgmsvs_scratch = sep_sgmsvs_scratch.T
        sep_sgmsvs_scratch_48k = resample(sep_sgmsvs_scratch, orig_sr=sr_sgmsvs_scratch, target_sr=48e3)
        
        sep_melroform_bigvgan, sr_melroform_bigvgan = soundfile.read(sep_file_melroform_bigvgan)
        if not(two_ch_flag):
            assert len(sep_melroform_bigvgan.shape) == 1, "Separated file is not mono!"
            sep_melroform_bigvgan = sep_melroform_bigvgan[:,None]
        sep_melroform_bigvgan = sep_melroform_bigvgan.T
        if sep_melroform_bigvgan.shape[1]>target.shape[1]:
            sep_melroform_bigvgan = sep_melroform_bigvgan[:,:target.shape[1]]
        sep_melroform_bigvgan_48k = resample(sep_melroform_bigvgan, orig_sr=sr_melroform_bigvgan, target_sr=48e3)
        
        sep_melroform_small, sr_melroform_small = soundfile.read(sep_file_melroform_small)
        if not(two_ch_flag):
            assert len(sep_melroform_small.shape) == 1, "Separated file is not mono!"
            sep_melroform_small = sep_melroform_small[:,None]
        sep_melroform_small = sep_melroform_small.T
        sep_melroform_small_48k = resample(sep_melroform_small, orig_sr=sr_melroform_small, target_sr=48e3)
        
        sep_melroform_large, sr_melroform_large = soundfile.read(sep_file_melroform_large)
        if not(two_ch_flag):
            assert len(sep_melroform_large.shape) == 1, "Separated file is not mono!"
            sep_melroform_large = sep_melroform_large[:,None]
        sep_melroform_large = sep_melroform_large.T
        sep_melroform_large_48k = resample(sep_melroform_large, orig_sr=sr_melroform_large, target_sr=48e3)
        
        sep_htdemucs, sr_htdemucs = soundfile.read(sep_file_htdemucs)
        if not(two_ch_flag):
            assert len(sep_htdemucs.shape) == 1, "Separated file is not mono!"
            sep_htdemucs = sep_htdemucs[:,None]
        sep_htdemucs = sep_htdemucs.T
        sep_htdemucs_48k = resample(sep_htdemucs, orig_sr=sr_htdemucs, target_sr=48e3)
        
        interference_folder = os.path.join(output_path,'tmp', 'interference')
        os.makedirs(interference_folder, exist_ok=True)
        interference = mixture - target
        interference_file = os.path.join(interference_folder, 'interference_'+file_id+'.wav')
        soundfile.write(interference_file, interference.T, args.sr)
        
       

        mixture = torch.from_numpy(mixture)
        target = torch.from_numpy(target)
        sep_sgmsvs_scratch = torch.from_numpy(sep_sgmsvs_scratch)
        sep_melroform_bigvgan = torch.from_numpy(sep_melroform_bigvgan)
        sep_melroform_small = torch.from_numpy(sep_melroform_small)
        sep_melroform_large = torch.from_numpy(sep_melroform_large)
        sep_htdemucs = torch.from_numpy(sep_htdemucs)
        
        mixture_48k = torch.from_numpy(mixture_48k)
        target_48k = torch.from_numpy(target_48k)
        sep_sgmsvs_scratch_48k = torch.from_numpy(sep_sgmsvs_scratch_48k)
        sep_melroform_bigvgan_48k = torch.from_numpy(sep_melroform_bigvgan_48k)
        sep_melroform_small_48k = torch.from_numpy(sep_melroform_small_48k)
        sep_melroform_large_48k = torch.from_numpy(sep_melroform_large_48k)
        sep_htdemucs_48k = torch.from_numpy(sep_htdemucs_48k)


        if CALCULATE_PEASS_BSS_EVAL_AND_MRES:
            destDir = os.path.join(MatDestDir, 'noisy', file_id)
            os.makedirs(destDir, exist_ok=True)
            MatOptions = {'destDir':destDir+os.path.sep, 'segmentationFactor':1}
            original_files = [target_file, interference_file]
            noisy_estimate_files = mixture_path
            noisy_peass_results = eng.PEASS_ObjectiveMeasure(original_files, noisy_estimate_files, MatOptions)
            _, _, tmp_sar = fast_bss_eval.bss_eval_sources(target, mixture, load_diag=REG_CONST, compute_permutation=PERM_FLAG, filter_length=FILT_LEN)
            multi_temp = []
            multi_mel_temp = []
            for ch in range(mixture.shape[0]):
                multi_temp.append(multi_res_loss(mixture[ch,...].float().unsqueeze(0).unsqueeze(0), target[ch,...].float().unsqueeze(0).unsqueeze(0)))
            multi_res_loss_scores_noisy.append(np.mean(multi_temp))
            
            sdr_scores_noisy.append(sdr(mixture, target))
            si_sdr_scores_noisy.append(si_sdr(mixture, target))
            sar_scores_noisy.append(tmp_sar)
            sir_scores_noisy.append(noisy_peass_results['SIR'])
            isr_scores_noisy.append(noisy_peass_results['ISR'])

            ops_scores_noisy.append(noisy_peass_results['OPS'])
            tps_scores_noisy.append(noisy_peass_results['TPS'])
            ips_scores_noisy.append(noisy_peass_results['IPS'])
            aps_scores_noisy.append(noisy_peass_results['APS'])
                  
        if CALCULATE_SINGMOS_XLS_R:
            resample_audio = torch.from_numpy(resample(mixture.numpy(), orig_sr=args.sr, target_sr=16000))
            length = torch.tensor([resample_audio.shape[1], resample_audio.shape[1]])
            singmos_scores = singmos_predictor(resample_audio.float(), length)
            singmos_mean = torch.mean(singmos_scores).item()
            singmos_scores_noisy.append(singmos_mean)
            
            xls_score = xls_model(resample_audio)
            xls_r_sqa_scores_noisy.append(xls_score.item())
            
        if CALCULATE_VISQOL:
            visqol_per_ch = []
            ref_temp = os.path.join(visqol_temp_folder, 'ref.wav')
            deg_temp = os.path.join(visqol_temp_folder, 'deg.wav')
            csv_temp = os.path.join(visqol_temp_folder, 'csv_temp.csv')
            for ch in range(mixture.shape[0]):
                soundfile.write(ref_temp, target_48k[ch].numpy().T.astype('float64'), 48000)
                soundfile.write(deg_temp, mixture_48k[ch].numpy().T.astype('float64'), 48000)
                visqol_return = subprocess.run([visqol_cl_api, '--reference_file', os.path.abspath(ref_temp), '--degraded_file', os.path.abspath(deg_temp), '--results_csv', os.path.abspath(csv_temp), '--similarity_to_quality_model', './model/libsvm_nu_svr_model.txt'], cwd='./visqol-master')
                moslqo = pd.read_csv(csv_temp)['moslqo'].loc[0]
                visqol_per_ch.append(moslqo)
            visqol_scores_noisy.append(np.mean(visqol_per_ch))
            shutil.rmtree(visqol_temp_folder)
            os.makedirs(visqol_temp_folder, exist_ok=True)
            
        if CALCULATE_AUDIOBOX:
            ab_aes = audiobox_predictor.forward([{"path":mixture.type(torch.float), "sample_rate":args.sr}])
            meta_aes_pq_scores_noisy.append(ab_aes[0]['PQ'])
            meta_aes_cu_scores_noisy.append(ab_aes[0]['CU'])



        if CALCULATE_PEASS_BSS_EVAL_AND_MRES:
            destDir = os.path.join(MatDestDir, 'sgmsvs_scratch', file_id)
            os.makedirs(destDir, exist_ok=True)
            MatOptions = {'destDir':destDir+os.path.sep, 'segmentationFactor':1}
            original_files = [target_file, interference_file]
            sgmsvs_scratch_estimate_files = sep_file_sgmsvs_from_scratch
            sgmsvs_scratch_peass_results = eng.PEASS_ObjectiveMeasure(original_files, sgmsvs_scratch_estimate_files, MatOptions)
            _, _, tmp_sar = fast_bss_eval.bss_eval_sources(target, sep_sgmsvs_scratch, load_diag=REG_CONST, compute_permutation=PERM_FLAG, filter_length=FILT_LEN)

            multi_temp = []
            multi_mel_temp = []
            for ch in range(sep_sgmsvs_scratch.shape[0]):
                multi_temp.append(multi_res_loss(sep_sgmsvs_scratch[ch,...].float().unsqueeze(0).unsqueeze(0), target[ch,...].float().unsqueeze(0).unsqueeze(0)))
            multi_res_loss_scores_sgmsvs_scratch.append(np.mean(multi_temp))

            sdr_scores_sgmsvs_scratch.append(sdr(sep_sgmsvs_scratch, target))
            si_sdr_scores_sgmsvs_scratch.append(si_sdr(sep_sgmsvs_scratch, target))
            sar_scores_sgmsvs_scratch.append(tmp_sar)
            sir_scores_sgmsvs_scratch.append(sgmsvs_scratch_peass_results['SIR'])
            isr_scores_sgmsvs_scratch.append(sgmsvs_scratch_peass_results['ISR'])
        
            ops_scores_sgmsvs_scratch.append(sgmsvs_scratch_peass_results['OPS'])
            tps_scores_sgmsvs_scratch.append(sgmsvs_scratch_peass_results['TPS'])
            ips_scores_sgmsvs_scratch.append(sgmsvs_scratch_peass_results['IPS'])
            aps_scores_sgmsvs_scratch.append(sgmsvs_scratch_peass_results['APS'])

        if CALCULATE_SINGMOS_XLS_R:
            resample_audio = torch.from_numpy(resample(sep_sgmsvs_scratch.numpy(), orig_sr=args.sr, target_sr=16000))
            length = torch.tensor([resample_audio.shape[1], resample_audio.shape[1]])
            singmos_scores = singmos_predictor(resample_audio.float(), length)      
            singmos_mean = torch.mean(singmos_scores).item()
            singmos_scores_sgmsvs_scratch.append(singmos_mean)
            xls_score = xls_model(resample_audio)
            xls_r_sqa_scores_sgmsvs_scratch.append(xls_score.item())
            
        if CALCULATE_VISQOL:
            visqol_per_ch = []
            ref_temp = os.path.join(visqol_temp_folder, 'ref.wav')
            deg_temp = os.path.join(visqol_temp_folder, 'deg.wav')
            csv_temp = os.path.join(visqol_temp_folder, 'csv_temp.csv')
            for ch in range(mixture.shape[0]):
                soundfile.write(ref_temp, target_48k[ch].numpy().T.astype('float64'), 48000)
                soundfile.write(deg_temp, sep_sgmsvs_scratch_48k[ch].numpy().T.astype('float64'), 48000)
                visqol_return = subprocess.run([visqol_cl_api, '--reference_file', os.path.abspath(ref_temp), '--degraded_file', os.path.abspath(deg_temp), '--results_csv', os.path.abspath(csv_temp), '--similarity_to_quality_model', './model/libsvm_nu_svr_model.txt'], cwd='./visqol-master')
                moslqo = pd.read_csv(csv_temp)['moslqo'].loc[0]
                visqol_per_ch.append(moslqo)
            visqol_scores_sgmsvs_scratch.append(np.mean(visqol_per_ch))
            shutil.rmtree(visqol_temp_folder)
            os.makedirs(visqol_temp_folder, exist_ok=True)


        if CALCULATE_AUDIOBOX:
            ab_aes = audiobox_predictor.forward([{"path":sep_sgmsvs_scratch.type(torch.float), "sample_rate":args.sr}])
            meta_aes_pq_scores_sgmsvs_scratch.append(ab_aes[0]['PQ'])
            meta_aes_cu_scores_sgmsvs_scratch.append(ab_aes[0]['CU'])
                       
  
        if CALCULATE_PEASS_BSS_EVAL_AND_MRES:
            destDir = os.path.join(MatDestDir, 'melroform_bigvgan', file_id)
            os.makedirs(destDir, exist_ok=True)
            os.makedirs(os.path.join(destDir,'tmp'), exist_ok=True)
            MatOptions = {'destDir':destDir+os.path.sep, 'segmentationFactor':1}
            original_files = [target_file, interference_file]
            melroform_bigvgan_estimate_files = sep_file_melroform_bigvgan
            tmp_file = os.path.join(destDir,'tmp','tmp.wav')
            sep_mel_roform_big_vgan_tmp, sr_tmp = soundfile.read(sep_file_melroform_bigvgan)
            target_len = soundfile.info(target_file).duration
            sep_mel_roform_big_vgan_tmp = sep_mel_roform_big_vgan_tmp[:int(target_len*sr_tmp)]
            soundfile.write(tmp_file, sep_mel_roform_big_vgan_tmp, sr_tmp)
            melroform_bigvgan_peass_results = eng.PEASS_ObjectiveMeasure(original_files, tmp_file, MatOptions)
            multi_temp = []
            multi_mel_temp = []
            for ch in range(sep_melroform_bigvgan.shape[0]):
                multi_temp.append(multi_res_loss(sep_melroform_bigvgan[ch,...].float().unsqueeze(0).unsqueeze(0), target[ch,...].float().unsqueeze(0).unsqueeze(0)))
            multi_res_loss_scores_melroform_bigvgan.append(np.mean(multi_temp))
            
            
            _, _, tmp_sar = fast_bss_eval.bss_eval_sources(target, sep_melroform_bigvgan, load_diag=REG_CONST, compute_permutation=PERM_FLAG, filter_length=FILT_LEN)
            sdr_scores_melroform_bigvgan.append(sdr(sep_melroform_bigvgan, target))
            si_sdr_scores_melroform_bigvgan.append(si_sdr(sep_melroform_bigvgan, target))
            sar_scores_melroform_bigvgan.append(tmp_sar)
            sir_scores_melroform_bigvgan.append(melroform_bigvgan_peass_results['SIR'])
            isr_scores_melroform_bigvgan.append(melroform_bigvgan_peass_results['ISR'])
            
            ops_scores_melroform_bigvgan.append(melroform_bigvgan_peass_results['OPS'])
            tps_scores_melroform_bigvgan.append(melroform_bigvgan_peass_results['TPS'])
            ips_scores_melroform_bigvgan.append(melroform_bigvgan_peass_results['IPS'])
            aps_scores_melroform_bigvgan.append(melroform_bigvgan_peass_results['APS'])
            os.remove(os.path.join(destDir,'tmp','tmp.wav'))
            os.rmdir(os.path.join(destDir,'tmp'))
            del sep_mel_roform_big_vgan_tmp, sr_tmp, tmp_file
    
            
            
        if CALCULATE_SINGMOS_XLS_R:
            resample_audio = torch.from_numpy(resample(sep_melroform_bigvgan.numpy(), orig_sr=args.sr, target_sr=16000))
            length = torch.tensor([resample_audio.shape[1], resample_audio.shape[1]])
            singmos_scores = singmos_predictor(resample_audio.float(), length)
            singmos_mean = torch.mean(singmos_scores).item()
            singmos_scores_melroform_bigvgan.append(singmos_mean)
            xls_score = xls_model(resample_audio)
            xls_r_sqa_scores_melroform_bigvgan.append(xls_score.item())
        
        if CALCULATE_VISQOL:
            visqol_per_ch = []
            ref_temp = os.path.join(visqol_temp_folder, 'ref.wav')
            deg_temp = os.path.join(visqol_temp_folder, 'deg.wav')
            csv_temp = os.path.join(visqol_temp_folder, 'csv_temp.csv')
            for ch in range(mixture.shape[0]):
                soundfile.write(ref_temp, target_48k[ch].numpy().T.astype('float64'), 48000)
                soundfile.write(deg_temp, sep_melroform_bigvgan_48k[ch].numpy().T.astype('float64'), 48000)
                visqol_return = subprocess.run([visqol_cl_api, '--reference_file', os.path.abspath(ref_temp), '--degraded_file', os.path.abspath(deg_temp), '--results_csv', os.path.abspath(csv_temp), '--similarity_to_quality_model', './model/libsvm_nu_svr_model.txt'], cwd='./visqol-master')
                moslqo = pd.read_csv(csv_temp)['moslqo'].loc[0]
                visqol_per_ch.append(moslqo)
            visqol_scores_melroform_bigvgan.append(np.mean(visqol_per_ch))
            shutil.rmtree(visqol_temp_folder)
            os.makedirs(visqol_temp_folder, exist_ok=True)

        if CALCULATE_AUDIOBOX:
            ab_aes = audiobox_predictor.forward([{"path":sep_melroform_bigvgan.type(torch.float), "sample_rate":args.sr}])
            meta_aes_pq_scores_melroform_bigvgan.append(ab_aes[0]['PQ'])
            meta_aes_cu_scores_melroform_bigvgan.append(ab_aes[0]['CU'])
            
        if CALCULATE_PEASS_BSS_EVAL_AND_MRES:        
            destDir = os.path.join(MatDestDir, 'melroform', file_id)
            os.makedirs(destDir, exist_ok=True)
            MatOptions = {'destDir':destDir+os.path.sep, 'segmentationFactor':1}
            original_files = [target_file, interference_file]
            melroform_estimate_files = sep_file_melroform_small
            melroform_peass_results = eng.PEASS_ObjectiveMeasure(original_files, melroform_estimate_files, MatOptions)
            _, _, tmp_sar = fast_bss_eval.bss_eval_sources(target, sep_melroform_small, load_diag=REG_CONST, compute_permutation=PERM_FLAG, filter_length=FILT_LEN)

            multi_temp = []
            multi_mel_temp = []
            for ch in range(sep_melroform_small.shape[0]):
                multi_temp.append(multi_res_loss(sep_melroform_small[ch,...].float().unsqueeze(0).unsqueeze(0), target[ch,...].float().unsqueeze(0).unsqueeze(0)))
            multi_res_loss_scores_melroform_small.append(np.mean(multi_temp))

            sdr_scores_melroform_small.append(sdr(sep_melroform_small, target))
            si_sdr_scores_melroform_small.append(si_sdr(sep_melroform_small, target))
            sar_scores_melroform_small.append(tmp_sar)
            sir_scores_melroform_small.append(melroform_peass_results['SIR'])
            isr_scores_melroform_small.append(melroform_peass_results['ISR'])

            ops_scores_melroform_small.append(melroform_peass_results['OPS'])
            tps_scores_melroform_small.append(melroform_peass_results['TPS'])
            ips_scores_melroform_small.append(melroform_peass_results['IPS'])
            aps_scores_melroform_small.append(melroform_peass_results['APS'])

        if CALCULATE_SINGMOS_XLS_R:
            resample_audio = torch.from_numpy(resample(sep_melroform_small.numpy(), orig_sr=args.sr, target_sr=16000))
            length = torch.tensor([resample_audio.shape[1], resample_audio.shape[1]])
            singmos_scores = singmos_predictor(resample_audio.float(),length)      
            singmos_mean = torch.mean(singmos_scores).item()
            singmos_scores_melroform_small.append(singmos_mean)
            xls_score = xls_model(resample_audio)
            xls_r_sqa_scores_melroform_small.append(xls_score.item())
            
        if CALCULATE_VISQOL:
            visqol_per_ch = []
            ref_temp = os.path.join(visqol_temp_folder, 'ref.wav')
            deg_temp = os.path.join(visqol_temp_folder, 'deg.wav')
            csv_temp = os.path.join(visqol_temp_folder, 'csv_temp.csv')
            for ch in range(mixture.shape[0]):
                soundfile.write(ref_temp, target_48k[ch].numpy().T.astype('float64'), 48000)
                soundfile.write(deg_temp, sep_melroform_small_48k[ch].numpy().T.astype('float64'), 48000)
                visqol_return = subprocess.run([visqol_cl_api, '--reference_file', os.path.abspath(ref_temp), '--degraded_file', os.path.abspath(deg_temp), '--results_csv', os.path.abspath(csv_temp), '--similarity_to_quality_model', './model/libsvm_nu_svr_model.txt'], cwd='./visqol-master')
                moslqo = pd.read_csv(csv_temp)['moslqo'].loc[0]
                visqol_per_ch.append(moslqo)
            visqol_scores_melroform_small.append(np.mean(visqol_per_ch))
            shutil.rmtree(visqol_temp_folder)
            os.makedirs(visqol_temp_folder, exist_ok=True)

        if CALCULATE_AUDIOBOX:
            ab_aes = audiobox_predictor.forward([{"path":sep_melroform_small.type(torch.float), "sample_rate":args.sr}])
            meta_aes_pq_scores_melroform_small.append(ab_aes[0]['PQ'])
            meta_aes_cu_scores_melroform_small.append(ab_aes[0]['CU'])
            

        if CALCULATE_PEASS_BSS_EVAL_AND_MRES:        
            destDir = os.path.join(MatDestDir, 'melroform', file_id)
            os.makedirs(destDir, exist_ok=True)
            MatOptions = {'destDir':destDir+os.path.sep, 'segmentationFactor':1}
            original_files = [target_file, interference_file]
            melroform_estimate_files = sep_file_melroform_large
            melroform_peass_results = eng.PEASS_ObjectiveMeasure(original_files, melroform_estimate_files, MatOptions)
            _, _, tmp_sar = fast_bss_eval.bss_eval_sources(target, sep_melroform_large, load_diag=REG_CONST, compute_permutation=PERM_FLAG, filter_length=FILT_LEN)

            multi_temp = []
            multi_mel_temp = []
            for ch in range(sep_melroform_large.shape[0]):
                multi_temp.append(multi_res_loss(sep_melroform_large[ch,...].float().unsqueeze(0).unsqueeze(0), target[ch,...].float().unsqueeze(0).unsqueeze(0)))
            multi_res_loss_scores_melroform_large.append(np.mean(multi_temp))

            sdr_scores_melroform_large.append(sdr(sep_melroform_large, target))
            si_sdr_scores_melroform_large.append(si_sdr(sep_melroform_large, target))
            sar_scores_melroform_large.append(tmp_sar)
            sir_scores_melroform_large.append(melroform_peass_results['SIR'])
            isr_scores_melroform_large.append(melroform_peass_results['ISR'])

            ops_scores_melroform_large.append(melroform_peass_results['OPS'])
            tps_scores_melroform_large.append(melroform_peass_results['TPS'])
            ips_scores_melroform_large.append(melroform_peass_results['IPS'])
            aps_scores_melroform_large.append(melroform_peass_results['APS'])

        if CALCULATE_SINGMOS_XLS_R:
            resample_audio = torch.from_numpy(resample(sep_melroform_large.numpy(), orig_sr=args.sr, target_sr=16000))
            length = torch.tensor([resample_audio.shape[1], resample_audio.shape[1]])
            singmos_scores = singmos_predictor(resample_audio.float(),length)      
            singmos_mean = torch.mean(singmos_scores).item()
            singmos_scores_melroform_large.append(singmos_mean)
            xls_score = xls_model(resample_audio)
            xls_r_sqa_scores_melroform_large.append(xls_score.item())
            
        if CALCULATE_VISQOL:
            visqol_per_ch = []
            ref_temp = os.path.join(visqol_temp_folder, 'ref.wav')
            deg_temp = os.path.join(visqol_temp_folder, 'deg.wav')
            csv_temp = os.path.join(visqol_temp_folder, 'csv_temp.csv')
            for ch in range(mixture.shape[0]):
                soundfile.write(ref_temp, target_48k[ch].numpy().T.astype('float64'), 48000)
                soundfile.write(deg_temp, sep_melroform_large_48k[ch].numpy().T.astype('float64'), 48000)
                visqol_return = subprocess.run([visqol_cl_api, '--reference_file', os.path.abspath(ref_temp), '--degraded_file', os.path.abspath(deg_temp), '--results_csv', os.path.abspath(csv_temp), '--similarity_to_quality_model', './model/libsvm_nu_svr_model.txt'], cwd='./visqol-master')
                moslqo = pd.read_csv(csv_temp)['moslqo'].loc[0]
                visqol_per_ch.append(moslqo)
            visqol_scores_melroform_large.append(np.mean(visqol_per_ch))
            shutil.rmtree(visqol_temp_folder)
            os.makedirs(visqol_temp_folder, exist_ok=True)
            
        if CALCULATE_AUDIOBOX:
            ab_aes = audiobox_predictor.forward([{"path":sep_melroform_large.type(torch.float), "sample_rate":args.sr}])
            meta_aes_pq_scores_melroform_large.append(ab_aes[0]['PQ'])
            meta_aes_cu_scores_melroform_large.append(ab_aes[0]['CU'])

        if CALCULATE_PEASS_BSS_EVAL_AND_MRES:
            destDir = os.path.join(MatDestDir, 'htdemucs', file_id)
            os.makedirs(destDir, exist_ok=True) 
            MatOptions = {'destDir':destDir+os.path.sep, 'segmentationFactor':1}
            original_files = [target_file, interference_file]
            htdemucs_estimate_files = sep_file_htdemucs
            htdemucs_peass_results = eng.PEASS_ObjectiveMeasure(original_files, htdemucs_estimate_files, MatOptions)
            _, _, tmp_sar = fast_bss_eval.bss_eval_sources(target, sep_htdemucs, load_diag=REG_CONST, compute_permutation=PERM_FLAG, filter_length=FILT_LEN)

            multi_temp = []
            multi_mel_temp = []
            for ch in range(sep_htdemucs.shape[0]):
                multi_temp.append(multi_res_loss(sep_htdemucs[ch,...].float().unsqueeze(0).unsqueeze(0), target[ch,...].float().unsqueeze(0).unsqueeze(0)))
            multi_res_loss_scores_htdemucs.append(np.mean(multi_temp))

            sdr_scores_htdemucs.append(sdr(sep_htdemucs, target))
            si_sdr_scores_htdemucs.append(si_sdr(sep_htdemucs, target))
            sar_scores_htdemucs.append(tmp_sar)
            sir_scores_htdemucs.append(htdemucs_peass_results['SIR'])
            isr_scores_htdemucs.append(htdemucs_peass_results['ISR'])

            ops_scores_htdemucs.append(htdemucs_peass_results['OPS'])
            tps_scores_htdemucs.append(htdemucs_peass_results['TPS'])
            ips_scores_htdemucs.append(htdemucs_peass_results['IPS'])
            aps_scores_htdemucs.append(htdemucs_peass_results['APS'])

        if CALCULATE_SINGMOS_XLS_R:
            resample_audio = torch.from_numpy(resample(sep_htdemucs.numpy(), orig_sr=args.sr, target_sr=16000))
            length = torch.tensor([resample_audio.shape[1], resample_audio.shape[1]])
            singmos_scores = singmos_predictor(resample_audio.float(), length)     
            singmos_mean = torch.mean(singmos_scores).item()
            singmos_scores_htdemucs.append(singmos_mean)
            xls_score = xls_model(resample_audio)
            xls_r_sqa_scores_htdemucs.append(xls_score.item())
            
        if CALCULATE_VISQOL:
            visqol_per_ch = []
            ref_temp = os.path.join(visqol_temp_folder, 'ref.wav')
            deg_temp = os.path.join(visqol_temp_folder, 'deg.wav')
            csv_temp = os.path.join(visqol_temp_folder, 'csv_temp.csv')
            for ch in range(mixture.shape[0]):
                soundfile.write(ref_temp, target_48k[ch].numpy().T.astype('float64'), 48000)
                soundfile.write(deg_temp, sep_htdemucs_48k[ch].numpy().T.astype('float64'), 48000)
                visqol_return = subprocess.run([visqol_cl_api, '--reference_file', os.path.abspath(ref_temp), '--degraded_file', os.path.abspath(deg_temp), '--results_csv', os.path.abspath(csv_temp), '--similarity_to_quality_model', './model/libsvm_nu_svr_model.txt'], cwd='./visqol-master')
                moslqo = pd.read_csv(csv_temp)['moslqo'].loc[0]
                visqol_per_ch.append(moslqo)
            visqol_scores_htdemucs.append(np.mean(visqol_per_ch))
            shutil.rmtree(visqol_temp_folder)
            os.makedirs(visqol_temp_folder, exist_ok=True)

           
        if CALCULATE_AUDIOBOX:
            ab_aes = audiobox_predictor.forward([{"path":sep_htdemucs.type(torch.float), "sample_rate":args.sr}])
            meta_aes_pq_scores_htdemucs.append(ab_aes[0]['PQ'])
            meta_aes_cu_scores_htdemucs.append(ab_aes[0]['CU'])
            
    sdr_scores_noisy = np.array(sdr_scores_noisy)
    si_sdr_scores_noisy = np.array(si_sdr_scores_noisy)
    multi_res_loss_scores_noisy = np.array(multi_res_loss_scores_noisy)
    sir_scores_noisy = np.array(sir_scores_noisy)
    sar_scores_noisy = np.array(sar_scores_noisy)
    ops_scores_noisy = np.array(ops_scores_noisy)
    tps_scores_noisy = np.array(tps_scores_noisy)
    ips_scores_noisy = np.array(ips_scores_noisy)
    aps_scores_noisy = np.array(aps_scores_noisy)
    singmos_scores_noisy = np.array(singmos_scores_noisy)
    visqol_scores_noisy = np.array(visqol_scores_noisy)
    meta_aes_pq_scores_noisy = np.array(meta_aes_pq_scores_noisy)
    meta_aes_cu_scores_noisy = np.array(meta_aes_cu_scores_noisy)
    xls_r_sqa_scores_noisy = np.array(xls_r_sqa_scores_noisy)

    sdr_scores_sgmsvs_scratch = np.array(sdr_scores_sgmsvs_scratch)
    si_sdr_scores_sgmsvs_scratch = np.array(si_sdr_scores_sgmsvs_scratch)
    multi_res_loss_scores_sgmsvs_scratch = np.array(multi_res_loss_scores_sgmsvs_scratch)
    sir_scores_sgmsvs_scratch = np.array(sir_scores_sgmsvs_scratch)
    sar_scores_sgmsvs_scratch = np.array(sar_scores_sgmsvs_scratch)
    ops_scores_sgmsvs_scratch = np.array(ops_scores_sgmsvs_scratch)
    tps_scores_sgmsvs_scratch = np.array(tps_scores_sgmsvs_scratch)
    ips_scores_sgmsvs_scratch = np.array(ips_scores_sgmsvs_scratch)
    aps_scores_sgmsvs_scratch = np.array(aps_scores_sgmsvs_scratch)
    singmos_scores_sgmsvs_scratch = np.array(singmos_scores_sgmsvs_scratch)
    visqol_scores_sgmsvs_scratch = np.array(visqol_scores_sgmsvs_scratch)
    meta_aes_pq_scores_sgmsvs_scratch = np.array(meta_aes_pq_scores_sgmsvs_scratch)
    meta_aes_cu_scores_sgmsvs_scratch = np.array(meta_aes_cu_scores_sgmsvs_scratch)
    xls_r_sqa_scores_sgmsvs_scratch = np.array(xls_r_sqa_scores_sgmsvs_scratch)
    
    sdr_scores_melroform_bigvgan = np.array(sdr_scores_melroform_bigvgan)
    si_sdr_scores_melroform_bigvgan = np.array(si_sdr_scores_melroform_bigvgan)
    multi_res_loss_scores_melroform_bigvgan = np.array(multi_res_loss_scores_melroform_bigvgan)
    sir_scores_melroform_bigvgan = np.array(sir_scores_melroform_bigvgan)
    sar_scores_melroform_bigvgan = np.array(sar_scores_melroform_bigvgan)
    ops_scores_melroform_bigvgan = np.array(ops_scores_melroform_bigvgan)
    tps_scores_melroform_bigvgan = np.array(tps_scores_melroform_bigvgan)
    ips_scores_melroform_bigvgan = np.array(ips_scores_melroform_bigvgan)
    aps_scores_melroform_bigvgan = np.array(aps_scores_melroform_bigvgan)
    singmos_scores_melroform_bigvgan = np.array(singmos_scores_melroform_bigvgan)
    visqol_scores_melroform_bigvgan = np.array(visqol_scores_melroform_bigvgan)
    meta_aes_pq_scores_melroform_bigvgan = np.array(meta_aes_pq_scores_melroform_bigvgan)
    meta_aes_cu_scores_melroform_bigvgan = np.array(meta_aes_cu_scores_melroform_bigvgan)
    xls_r_sqa_scores_melroform_bigvgan = np.array(xls_r_sqa_scores_melroform_bigvgan)

    sdr_scores_melroform_small = np.array(sdr_scores_melroform_small)
    si_sdr_scores_melroform_small = np.array(si_sdr_scores_melroform_small)
    multi_res_loss_scores_melroform_small = np.array(multi_res_loss_scores_melroform_small)
    sir_scores_melroform_small = np.array(sir_scores_melroform_small)
    sar_scores_melroform_small = np.array(sar_scores_melroform_small)
    ops_scores_melroform_small = np.array(ops_scores_melroform_small)
    tps_scores_melroform_small = np.array(tps_scores_melroform_small)
    ips_scores_melroform_small = np.array(ips_scores_melroform_small)
    aps_scores_melroform_small = np.array(aps_scores_melroform_small)
    singmos_scores_melroform_small = np.array(singmos_scores_melroform_small)
    visqol_scores_melroform_small = np.array(visqol_scores_melroform_small)
    meta_aes_pq_scores_melroform_small = np.array(meta_aes_pq_scores_melroform_small)
    meta_aes_cu_scores_melroform_small = np.array(meta_aes_cu_scores_melroform_small)
    xls_r_sqa_scores_melroform_small = np.array(xls_r_sqa_scores_melroform_small)

    sdr_scores_melroform_large = np.array(sdr_scores_melroform_large)
    si_sdr_scores_melroform_large = np.array(si_sdr_scores_melroform_large)
    multi_res_loss_scores_melroform_large = np.array(multi_res_loss_scores_melroform_large)
    sir_scores_melroform_large = np.array(sir_scores_melroform_large)
    sar_scores_melroform_large = np.array(sar_scores_melroform_large)
    ops_scores_melroform_large = np.array(ops_scores_melroform_large)
    tps_scores_melroform_large = np.array(tps_scores_melroform_large)
    ips_scores_melroform_large = np.array(ips_scores_melroform_large)
    aps_scores_melroform_large = np.array(aps_scores_melroform_large)
    singmos_scores_melroform_large = np.array(singmos_scores_melroform_large)
    visqol_scores_melroform_large = np.array(visqol_scores_melroform_large)
    meta_aes_pq_scores_melroform_large = np.array(meta_aes_pq_scores_melroform_large)
    meta_aes_cu_scores_melroform_large = np.array(meta_aes_cu_scores_melroform_large)
    xls_r_sqa_scores_melroform_large = np.array(xls_r_sqa_scores_melroform_large)

    sdr_scores_htdemucs = np.array(sdr_scores_htdemucs)
    si_sdr_scores_htdemucs = np.array(si_sdr_scores_htdemucs)
    multi_res_loss_scores_htdemucs = np.array(multi_res_loss_scores_htdemucs)
    sir_scores_htdemucs = np.array(sir_scores_htdemucs)
    sar_scores_htdemucs = np.array(sar_scores_htdemucs)
    ops_scores_htdemucs = np.array(ops_scores_htdemucs)
    tps_scores_htdemucs = np.array(tps_scores_htdemucs)
    ips_scores_htdemucs = np.array(ips_scores_htdemucs)
    aps_scores_htdemucs = np.array(aps_scores_htdemucs)
    singmos_scores_htdemucs = np.array(singmos_scores_htdemucs)
    visqol_scores_htdemucs = np.array(visqol_scores_htdemucs)
    meta_aes_pq_scores_htdemucs = np.array(meta_aes_pq_scores_htdemucs)
    meta_aes_cu_scores_htdemucs = np.array(meta_aes_cu_scores_htdemucs)

    xls_r_sqa_scores_htdemucs = np.array(xls_r_sqa_scores_htdemucs)

    row_names = ['noisy', 'sgmsvs', 'melroformer_bigvgan', 'melroformer_small', 'melroformer_large', 'htdemucs']

    if CALCULATE_PEASS_BSS_EVAL_AND_MRES:
        if len(sdr_scores_noisy.shape)>1:
            sdr_data = np.stack((np.mean(sdr_scores_noisy,1), np.mean(sdr_scores_sgmsvs_scratch,1), np.mean(sdr_scores_melroform_bigvgan,1), np.mean(sdr_scores_melroform_small,1), np.mean(sdr_scores_melroform_large,1),  np.mean(sdr_scores_htdemucs,1)), axis=1)
            sisdr_data = np.stack((np.mean(si_sdr_scores_noisy,1), np.mean(si_sdr_scores_sgmsvs_scratch,1), np.mean(si_sdr_scores_melroform_bigvgan,1), np.mean(si_sdr_scores_melroform_small,1), np.mean(si_sdr_scores_melroform_large,1), np.mean(si_sdr_scores_htdemucs,1)), axis=1)
            sar_data = np.stack((np.mean(sar_scores_noisy,1), np.mean(sar_scores_sgmsvs_scratch,1), np.mean(sar_scores_melroform_bigvgan,1), np.mean(sar_scores_melroform_small,1), np.mean(sar_scores_melroform_large,1), np.mean(sar_scores_htdemucs,1)), axis=1)
        else:
            #no mean
            sdr_data = np.stack((sdr_scores_noisy, sdr_scores_sgmsvs_scratch, sdr_scores_melroform_bigvgan, sdr_scores_melroform_small, sdr_scores_melroform_large,  sdr_scores_htdemucs), axis=1)
            sisdr_data = np.stack((np.mean(si_sdr_scores_noisy,1), np.mean(si_sdr_scores_sgmsvs_scratch,1), np.mean(si_sdr_scores_melroform_bigvgan,1), np.mean(si_sdr_scores_melroform_small,1), np.mean(si_sdr_scores_melroform_large,1), np.mean(si_sdr_scores_htdemucs,1)), axis=1)
            sar_data = np.stack((sar_scores_noisy, sar_scores_sgmsvs_scratch, sar_scores_melroform_bigvgan, sar_scores_melroform_small, sar_scores_melroform_large, sar_scores_htdemucs), axis=1)
        isr_data = np.stack((isr_scores_noisy, isr_scores_sgmsvs_scratch, isr_scores_melroform_bigvgan, isr_scores_melroform_small, isr_scores_melroform_large, isr_scores_htdemucs), axis=1)
        sir_data = np.stack((sir_scores_noisy, sir_scores_sgmsvs_scratch, sir_scores_melroform_bigvgan, sir_scores_melroform_small, sir_scores_melroform_large, sir_scores_htdemucs), axis=1)

        pd_sdr = pd.DataFrame(sdr_data, columns=row_names)
        pd_sisdr = pd.DataFrame(sisdr_data, columns=row_names)
        pd_sir = pd.DataFrame(sir_data, columns=row_names)
        pd_sar = pd.DataFrame(sar_data, columns=row_names)
        pd_isr = pd.DataFrame(isr_data, columns=row_names)
        
        #concatenate file_order
        pd_sdr = pd.concat([pd_file_order, pd_sdr], axis=1)
        pd_sisdr = pd.concat([pd_file_order, pd_sisdr], axis=1)
        pd_sir = pd.concat([pd_file_order, pd_sir], axis=1)
        pd_sar = pd.concat([pd_file_order, pd_sar], axis=1)
        pd_isr = pd.concat([pd_file_order, pd_isr], axis=1)
        
        pd_sdr.to_csv(os.path.join(output_path, 'sdr.csv'))
        pd_sisdr.to_csv(os.path.join(output_path, 'sisdr.csv'))
        pd_sir.to_csv(os.path.join(output_path, 'sir.csv'))
        pd_sar.to_csv(os.path.join(output_path, 'sar.csv'))
        pd_isr.to_csv(os.path.join(output_path, 'isr.csv'))
        
        
#    if CALCULATE_PEASS:     
        ops_data = np.stack((ops_scores_noisy, ops_scores_sgmsvs_scratch, ops_scores_melroform_bigvgan, ops_scores_melroform_small, ops_scores_melroform_large, ops_scores_htdemucs), axis=1)
        tps_data = np.stack((tps_scores_noisy, tps_scores_sgmsvs_scratch, tps_scores_melroform_bigvgan, tps_scores_melroform_small, tps_scores_melroform_large, tps_scores_htdemucs), axis=1)
        ips_data = np.stack((ips_scores_noisy, ips_scores_sgmsvs_scratch, ips_scores_melroform_bigvgan, ips_scores_melroform_small, ips_scores_melroform_large, ips_scores_htdemucs), axis=1) 
        aps_data = np.stack((aps_scores_noisy, aps_scores_sgmsvs_scratch, aps_scores_melroform_bigvgan, aps_scores_melroform_small, aps_scores_melroform_large, aps_scores_htdemucs), axis=1)   
    
        pd_ops = pd.DataFrame(ops_data, columns=row_names)
        pd_tps = pd.DataFrame(tps_data, columns=row_names)
        pd_ips = pd.DataFrame(ips_data, columns=row_names)
        pd_aps = pd.DataFrame(aps_data, columns=row_names)
       
        #concatenate file_order
        pd_ops = pd.concat([pd_file_order, pd_ops], axis=1)
        pd_tps = pd.concat([pd_file_order, pd_tps], axis=1)
        pd_ips = pd.concat([pd_file_order, pd_ips], axis=1)
        pd_aps = pd.concat([pd_file_order, pd_aps], axis=1)

        pd_ops.to_csv(os.path.join(output_path, 'ops.csv'))
        pd_tps.to_csv(os.path.join(output_path, 'tps.csv'))
        pd_ips.to_csv(os.path.join(output_path, 'ips.csv'))
        pd_aps.to_csv(os.path.join(output_path, 'aps.csv'))
        
        
        mres_data = np.stack((multi_res_loss_scores_noisy, multi_res_loss_scores_sgmsvs_scratch, multi_res_loss_scores_melroform_bigvgan, multi_res_loss_scores_melroform_small, multi_res_loss_scores_melroform_large, multi_res_loss_scores_htdemucs), axis=1)
        pd_mres = pd.DataFrame(mres_data, columns=row_names)
        #concatenate file_order
        pd_mres = pd.concat([pd_file_order, pd_mres], axis=1)
        pd_mres.to_csv(os.path.join(output_path, 'multi_res_loss.csv'))
        
    if CALCULATE_SINGMOS_XLS_R:   
        singmos_data = np.stack((singmos_scores_noisy, singmos_scores_sgmsvs_scratch, singmos_scores_melroform_bigvgan, singmos_scores_melroform_small, singmos_scores_melroform_large, singmos_scores_htdemucs), axis=1)
        
        pd_singmos = pd.DataFrame(singmos_data, columns=row_names)
        #concatenate file_order df
        pd_singmos = pd.concat([pd_file_order, pd_singmos], axis=1)
        pd_singmos.to_csv(os.path.join(output_path, 'singmos.csv'))

        
        
        xls_r_sqa_data = np.stack((xls_r_sqa_scores_noisy, xls_r_sqa_scores_sgmsvs_scratch, xls_r_sqa_scores_melroform_bigvgan, xls_r_sqa_scores_melroform_small, xls_r_sqa_scores_melroform_large, xls_r_sqa_scores_htdemucs), axis=1)
        pd_xls_r_sqa = pd.DataFrame(xls_r_sqa_data, columns=row_names)
        #concatenate file_order df
        pd_xls_r_sqa = pd.concat([pd_file_order, pd_xls_r_sqa], axis=1)
        pd_xls_r_sqa.to_csv(os.path.join(output_path, 'xls_r_sqa.csv'))


    if CALCULATE_VISQOL:
        visqol_data = np.stack((visqol_scores_noisy, visqol_scores_sgmsvs_scratch, visqol_scores_melroform_bigvgan, visqol_scores_melroform_small, visqol_scores_melroform_large, visqol_scores_htdemucs), axis=1)
        pd_visqol = pd.DataFrame(visqol_data, columns=row_names)
      
        #concatenate file_order df
        pd_visqol = pd.concat([pd_file_order, pd_visqol], axis=1)
   
        pd_visqol.to_csv(os.path.join(output_path, 'visqol.csv'))
        
    if CALCULATE_AUDIOBOX:
        audiobox_data_pq = np.stack((meta_aes_pq_scores_noisy, meta_aes_pq_scores_sgmsvs_scratch, meta_aes_pq_scores_melroform_bigvgan, meta_aes_pq_scores_melroform_small, meta_aes_pq_scores_melroform_large, meta_aes_pq_scores_htdemucs), axis=1)
        audiobox_data_cu = np.stack((meta_aes_cu_scores_noisy, meta_aes_cu_scores_sgmsvs_scratch, meta_aes_cu_scores_melroform_bigvgan, meta_aes_cu_scores_melroform_small, meta_aes_cu_scores_melroform_large, meta_aes_cu_scores_htdemucs), axis=1)
        
        pd_audiobox_pq = pd.DataFrame(audiobox_data_pq, columns=row_names)
        pd_audiobox_cu = pd.DataFrame(audiobox_data_cu, columns=row_names)

        #concatenate file_order df
        pd_audiobox_pq = pd.concat([pd_file_order, pd_audiobox_pq], axis=1)
        pd_audiobox_cu = pd.concat([pd_file_order, pd_audiobox_cu], axis=1)
        
        
        pd_audiobox_pq.to_csv(os.path.join(output_path, 'meta_audiobox_aes_PQ.csv'))
        pd_audiobox_cu.to_csv(os.path.join(output_path, 'meta_audiobox_aes_CU.csv'))
       
    shutil.rmtree(os.path.join(output_path, 'tmp'))

