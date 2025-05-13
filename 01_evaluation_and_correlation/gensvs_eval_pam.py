import argparse
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pam_eval.PAM import PAM
from pam_eval.dataset import ExampleDatasetFolder
EVAL_FILE_ORDER = './01_evaluation_and_correlation/file_id_order.csv'
RESULTS_PATH = './01_evaluation_and_correlation/evaluation_metrics'
os.makedirs(RESULTS_PATH, exist_ok=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PAM")
    parser.add_argument('--mixture_dir', type=str, help='Folder path to evaluate')
    parser.add_argument('--separated_dir_sgmsvs_from_scratch', type=str, help='Folder path to data from sgmsvs model')
    parser.add_argument('--separated_dir_melroform_bigvgan', type=str, help='Folder path to data from melroformer+BigVGAN model')
    parser.add_argument('--separated_dir_melroform_small', type=str, help='Folder path to data from melroformer_small model')
    parser.add_argument('--separated_dir_melroform_large', type=str, help='Folder path to data from melroformer_large model')
    parser.add_argument('--separated_dir_htdemucs', type=str, help='Folder path to data from htdemucs model')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of examples per batch')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    args = parser.parse_args()

    # initialize PAM
    pam = PAM(use_cuda=torch.cuda.is_available())

    # Create a dictionary to store PAM scores for each dataset
    pam_scores_dict = {}

    # Define a function to process each dataset
    def process_dataset(dataset_dir, dataset_name):
        dataset = ExampleDatasetFolder(src=dataset_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=False, drop_last=False,
                                collate_fn=dataset.collate)
        dataset_pam_dict = {}
        for files, audios, sample_index in tqdm(dataloader, desc=f"Processing {dataset_name}"):
            file_ids = ['fileid' + file.split(os.path.sep)[-1].split('.wav')[0].split('fileid')[-1] for file in files]
            pam_scores, pam_segment_score = pam.evaluate(audios, sample_index)
            for file_id, pam_score in zip(file_ids, pam_scores):
                 dataset_pam_dict[file_id]=pam_score
        pam_scores_dict[dataset_name] = dataset_pam_dict

    # Process each dataset
    if args.mixture_dir:
        process_dataset(args.mixture_dir, "mixture")
    if args.separated_dir_sgmsvs_from_scratch:
        process_dataset(args.separated_dir_sgmsvs_from_scratch, "sgmsvs")
    if args.separated_dir_melroform_bigvgan:
        process_dataset(args.separated_dir_melroform_bigvgan, "melroform_bigvgan")
    if args.separated_dir_melroform_small:
        process_dataset(args.separated_dir_melroform_small, "melroform_small")
    if args.separated_dir_melroform_large:
        process_dataset(args.separated_dir_melroform_large, "melroform_large")
    if args.separated_dir_htdemucs:
        process_dataset(args.separated_dir_htdemucs, "htdemucs")

    pam_scores_dataframe = pd.DataFrame.from_dict(pam_scores_dict, orient='index').T
    #sort the dataframe by the order of the files in eval_file_order
    eval_file_order = pd.read_csv(EVAL_FILE_ORDER).drop(columns=['Unnamed: 0'])
    pam_scores_dataframe = pam_scores_dataframe.loc[eval_file_order['file_id']].reset_index(drop=False,names='file_id')
    pam_scores_dataframe.to_csv(os.path.join(RESULTS_PATH,'pam.csv'), index=True)


