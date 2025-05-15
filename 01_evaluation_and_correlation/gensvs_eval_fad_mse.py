import os
import pandas as pd
import shutil
from fadtk import cache_embedding_files
from fadtk_mod.fad_mod import FrechetAudioDistance
from fadtk_mod.model_loader_mod import *
from pathlib import Path
from glob import glob
from argparse import ArgumentParser

FILE_ORDER =  './01_evaluation_and_correlation/file_id_order.csv'

WORKERS = 8
models = {m.name: m for m in get_all_models()}

#TODO: checkout why mono processing is not working! is cg_iter needed? => check if you can replicate sdr compuation from torchmetrics

def sort_df_by_order_df(df, file_order_df):
    """
    Sorts a dataframe by the order of another dataframe according to file_ids.
    """
    #get fileids from path for each emb file
    for row in df.iterrows():
        file_id = row[1][0].split(os.path.sep)[-1].split('fileid_')[-1].split('.wav')[0]
        file_id = 'fileid_'+file_id
        #add new column with file_id to dataframe
        df.at[row[0], 'file_id'] = file_id
    #sort metrics by file_ids according to order of file_order
    # Create a categorical type with the desired order
    df['file_id'] = pd.Categorical(df['file_id'], categories=file_order_df['file_id'], ordered=True)
    # Sort based on that
    df = df.sort_values('file_id', ignore_index=True) 
    return df

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
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder,'fad_mse_temp'), exist_ok=True)
    
    file_order = pd.read_csv(FILE_ORDER, usecols=[1])                                     
    
    mixture_files = sorted(glob(os.path.join(args.mixture_dir, '*.wav')))
    target_files = sorted(glob(os.path.join(args.target_dir, '*.wav')))
    sep_files_sgmsvs_from_scratch = sorted(glob(os.path.join(args.separated_dir_sgmsvs_from_scratch, '*.wav')))
    sep_files_melroform_bigvgan = sorted(glob(os.path.join(args.separated_dir_melroform_bigvgan, '*.wav')))
    sep_files_melroform_small = sorted(glob(os.path.join(args.separated_dir_melroform_small, '*.wav')))
    sep_files_melroform_large = sorted(glob(os.path.join(args.separated_dir_melroform_large, '*.wav')))
    sep_files_htdemucs = sorted(glob(os.path.join(args.separated_dir_htdemucs, '*.wav')))

    per_song_fad_scores_noisy = []
    per_song_fad_scores_melroform_bigvgan = []
    per_song_fad_scores_sgmsvs_scratch = []
    per_song_fad_scores_melroform_small = []
    per_song_fad_scores_melroform_large = []

    ref_files = args.target_dir
    
    eval_file_list = [args.mixture_dir, args.separated_dir_sgmsvs_from_scratch, args.separated_dir_melroform_bigvgan, args.separated_dir_melroform_small, args.separated_dir_melroform_large, args.separated_dir_htdemucs]
    #TODO rewrite scores to csv file in same style as other metrics 
    emb_mse_clap_audio_df = pd.DataFrame([])
    emb_mse_clap_music_df = pd.DataFrame([])
    emb_mse_mert_df = pd.DataFrame([])
    emb_mse_music2latent_df = pd.DataFrame([])

    fad_song2song_clap_audio_df = pd.DataFrame([])
    fad_song2song_clap_music_df = pd.DataFrame([])
    fad_song2song_mert_df = pd.DataFrame([])
    fad_song2song_music2latent_df = pd.DataFrame([])

    for eval_files in eval_file_list: 
        fad_model_name = 'clap-laion-audio'
        model = models[fad_model_name]
        model_name = eval_files.split(os.path.sep)[-1].split('_5s_new_single_channel')[0]
        if 'htdemucs' in model_name:
            model_name = 'htdemucs'
        # 1. Calculate embedding files for each dataset
        for d in [ref_files, eval_files]:
            if Path(d).is_dir():
                cache_embedding_files(d, model, workers=WORKERS)

        fad = FrechetAudioDistance(model, audio_load_worker=WORKERS, load_model=False)
        logger = logging.getLogger(__name__)

        csv_path_emb_mse = Path(os.path.join(args.output_folder,'fad_mse_temp', model_name+'_emb_mse_'+fad_model_name+'.csv'))
        csv_path_song2song = Path(os.path.join(args.output_folder,'fad_mse_temp', model_name+'_fad_song2song_'+fad_model_name+'.csv'))

        fad.mse_song2song(ref_files, eval_files, csv_path_emb_mse)
        fad.score_song2song(ref_files, eval_files, csv_path_song2song)

        #get column with ratings => column number 1
        emb_mse_temp = pd.read_csv(csv_path_emb_mse, header=None)
        emb_mse_temp = sort_df_by_order_df(emb_mse_temp, file_order)
        emb_mse_clap_audio_df[model_name] = emb_mse_temp[1].reset_index(drop=True)
        
        fad_song2song_temp = pd.read_csv(csv_path_song2song, header=None)
        fad_song2song_temp = sort_df_by_order_df(fad_song2song_temp, file_order)
        fad_song2song_clap_audio_df[model_name] = fad_song2song_temp[1].reset_index(drop=True)

        
        fad_model_name = 'clap-laion-music'
        model = models[fad_model_name]

        # 1. Calculate embedding files for each dataset
        for d in [ref_files, eval_files]:
            if Path(d).is_dir():
                cache_embedding_files(d, model, workers=WORKERS)

        fad = FrechetAudioDistance(model, audio_load_worker=WORKERS, load_model=False)

        csv_path_emb_mse = Path(os.path.join(args.output_folder,'fad_mse_temp', model_name+'_emb_mse_'+fad_model_name+'.csv'))
        csv_path_song2song = Path(os.path.join(args.output_folder,'fad_mse_temp', model_name+'_fad_song2song_'+fad_model_name+'.csv'))

        fad.mse_song2song(ref_files, eval_files, csv_path_emb_mse)
        fad.score_song2song(ref_files, eval_files, csv_path_song2song)
        
        #get column with ratings => column number 1
        emb_mse_temp = pd.read_csv(csv_path_emb_mse, header=None)
        emb_mse_temp = sort_df_by_order_df(emb_mse_temp, file_order)
        emb_mse_clap_music_df[model_name] = emb_mse_temp[1].reset_index(drop=True)
        
        fad_song2song_temp = pd.read_csv(csv_path_song2song, header=None)
        fad_song2song_temp = sort_df_by_order_df(fad_song2song_temp, file_order)
        fad_song2song_clap_music_df[model_name] = fad_song2song_temp[1].reset_index(drop=True)

        fad_model_name = 'MERT-v1-95M'
        model = models[fad_model_name]
        # 1. Calculate embedding files for each dataset
        for d in [ref_files, eval_files]:
            if Path(d).is_dir():
                cache_embedding_files(d, model, workers=WORKERS)

        fad = FrechetAudioDistance(model, audio_load_worker=WORKERS, load_model=False)

        csv_path_emb_mse = Path(os.path.join(args.output_folder,'fad_mse_temp', model_name+'_emb_mse_'+fad_model_name+'.csv'))
        csv_path_song2song = Path(os.path.join(args.output_folder,'fad_mse_temp', model_name+'_fad_song2song_'+fad_model_name+'.csv'))
                
        fad.mse_song2song(ref_files, eval_files, csv_path_emb_mse)
        fad.score_song2song(ref_files, eval_files, csv_path_song2song)
        
        #get column with ratings => column number 1
        emb_mse_temp = pd.read_csv(csv_path_emb_mse, header=None)
        emb_mse_temp = sort_df_by_order_df(emb_mse_temp, file_order)
        emb_mse_mert_df[model_name] = emb_mse_temp[1].reset_index(drop=True)

        fad_song2song_temp = pd.read_csv(csv_path_song2song, header=None)
        fad_song2song_temp = sort_df_by_order_df(fad_song2song_temp, file_order)
        fad_song2song_mert_df[model_name] = fad_song2song_temp[1].reset_index(drop=True)
        
        fad_model_name = 'music2latent'
        model = models[fad_model_name]

        # 1. Calculate embedding files for each dataset
        for d in [ref_files, eval_files]:
            if Path(d).is_dir():
                cache_embedding_files(d, model, workers=WORKERS)
                
        fad = FrechetAudioDistance(model, audio_load_worker=WORKERS, load_model=False)
        
        csv_path_emb_mse = Path(os.path.join(args.output_folder,'fad_mse_temp',model_name+'_emb_mse_'+fad_model_name+'.csv'))
        csv_path_song2song = Path(os.path.join(args.output_folder,'fad_mse_temp',model_name+'_fad_song2song_'+fad_model_name+'.csv'))
        
        fad.mse_song2song(ref_files, eval_files, csv_path_emb_mse)
        fad.score_song2song(ref_files, eval_files, csv_path_song2song)
        
        #get column with ratings => column number 1
        emb_mse_temp = pd.read_csv(csv_path_emb_mse, header=None)
        emb_mse_temp = sort_df_by_order_df(emb_mse_temp, file_order)
        emb_mse_music2latent_df[model_name] = emb_mse_temp[1].reset_index(drop=True)

        fad_song2song_temp = pd.read_csv(csv_path_song2song, header=None)
        fad_song2song_temp = sort_df_by_order_df(fad_song2song_temp, file_order)
        fad_song2song_music2latent_df[model_name] = fad_song2song_temp[1].reset_index(drop=True)


    #concatenate file_id column to each dataframe
    emb_mse_clap_audio_df.reset_index(drop=True, inplace=True)
    emb_mse_clap_audio_df = pd.concat([file_order, emb_mse_clap_audio_df], axis=1)
    emb_mse_clap_music_df.reset_index(drop=True, inplace=True)
    emb_mse_clap_music_df = pd.concat([file_order, emb_mse_clap_music_df], axis=1)
    emb_mse_mert_df.reset_index(drop=True, inplace=True)
    emb_mse_mert_df = pd.concat([file_order, emb_mse_mert_df], axis=1)
    emb_mse_music2latent_df.reset_index(drop=True, inplace=True)
    emb_mse_music2latent_df = pd.concat([file_order, emb_mse_music2latent_df], axis=1)
    
    
    fad_song2song_clap_audio_df.reset_index(drop=True, inplace=True)
    fad_song2song_clap_audio_df = pd.concat([file_order, fad_song2song_clap_audio_df], axis=1)
    fad_song2song_clap_music_df.reset_index(drop=True, inplace=True)
    fad_song2song_clap_music_df = pd.concat([file_order, fad_song2song_clap_music_df], axis=1)
    fad_song2song_mert_df.reset_index(drop=True, inplace=True)
    fad_song2song_mert_df = pd.concat([file_order, fad_song2song_mert_df], axis=1)
    fad_song2song_music2latent_df.reset_index(drop=True, inplace=True)
    fad_song2song_music2latent_df = pd.concat([file_order, fad_song2song_music2latent_df], axis=1)

    emb_mse_clap_audio_df.to_csv(os.path.join(args.output_folder,'emb_mse_clap_audio_df.csv'))
    emb_mse_clap_music_df.to_csv(os.path.join(args.output_folder,'emb_mse_clap_music_df.csv'))
    emb_mse_mert_df.to_csv(os.path.join(args.output_folder,'emb_mse_mert_df.csv'))
    emb_mse_music2latent_df.to_csv(os.path.join(args.output_folder,'emb_mse_music2latent_df.csv'))
     
    fad_song2song_clap_audio_df.to_csv(os.path.join(args.output_folder,'fad_song2song_clap_audio_df.csv'))
    fad_song2song_clap_music_df.to_csv(os.path.join(args.output_folder,'fad_song2song_clap_music_df.csv'))
    fad_song2song_mert_df.to_csv(os.path.join(args.output_folder,'fad_song2song_mert_df.csv'))
    fad_song2song_music2latent_df.to_csv(os.path.join(args.output_folder,'fad_song2song_music2latent_df.csv'))
    
    #remove temp folder
    shutil.rmtree(os.path.join(args.output_folder,'fad_mse_temp'))
    
