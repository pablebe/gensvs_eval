# Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation
<img src="./03_evaluation_data/figures/gen_disc_srcc_tradeoff.png" alt="Correlation Results" width="100%">  

This repository contains the code accompanying the WASPAA 2025 paper "Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation" by Paul A. Bereuter, Benjamin Stahl, Mark D. Plumbley and Alois Sontacchi.
## ℹ️ Further information
- Paper: [Preprint](https://arxiv.org/pdf/2507.11427)
- Website: [Companion Page](https://pablebe.github.io/gensvs_eval_companion_page/) 
- Data: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15911723.svg)](https://doi.org/10.5281/zenodo.15911723)
- More Code: [PyPi Package](https://pypi.org/project/gensvs/)
- Model Checkpoints: [Hugging Face](https://huggingface.co/collections/pablebe/gensvs-eval-model-checkpoints-687e1c967b43f867f34d6225)

## 📈 Benchmark you own metrics
A simple Python script to benchmark objective metrics on our DMOS data using the paper's correlation analysis is included in the Zenodo dataset available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15911723.svg)](https://doi.org/10.5281/zenodo.15911723).<br>
Instructions for benchmarking your own audio quality metrics are provided in the dataset’s Readme.md file.

## 🏃🏽‍♀️‍➡️ Model Inference and Embedding-based MSE Evaluation

If you want to use the proposed generative models (SGMSVS, MelRoFo (S) + BigVGAN) to separate singing voice from musical mixtures, or if you want to compute the embedding-based MSE of MERT or Music2Latent embeddings take a look at our [PyPi package](https://pypi.org/project/gensvs/).

## 🚀 Getting Started:
We have released a [PyPi package](https://pypi.org/project/gensvs/) that enables straightforward inference of the generative models (SGMSVS & MelRoFo (S)+BigVGAN) presented in the paper. It also facilitates the computation of the embedding-based mean squared error (MSE) for the MERT and Music2Latent embeddings, which exhibited the highest correlation with DMOS data in our study. Besides enabling easy model inference and embedding-based MSE computation (currently only for MERT and Music2Latent), this package can be used to run all the training code provided in this GitHub repository. Information on how to set up an environment to run the inference and training code can be found below (see I.). You will also find instructions on setting up conda environments to reproduce the computation of all the metrics included in our study and to carry out the correlation analysis outlined in the paper (see II., III.).
### I. ```gensvs_env```: Conda environment for easy model inference, computation of MSE on MERT/Music2Latent embeddings and model training 
1. Create conda environment: ```$ ./env_info/gensvs_env.yml```
2. Activate conda environment ```$ conda activate gensvs_env```
2. Install [gensvs](https://pypi.org/project/gensvs/) package: ```$ pip install gensvs```
3. [Alternative] Run Bash script that sets up conda environment and installs [gensvs](https://pypi.org/project/gensvs/) from root directory: [```$ ./env_info/setup_gensvs_env.sh```](https://github.com/pablebe/gensvs_eval/blob/main/env_info/setup_gensvs_env.sh)

### II. ```gensvs_eval_env```: Conda environment for computation of all non-embedding-based intrusive and non-intrusive metrics and for the correlation analysis
Set up the following environment for evaluating all non-intrusive metrics of the paper, as well as for plot and *.csv export.
1. Create conda environment: ```$ ./env_info/svs_eval_env.yml```
2. Activate conda environment: ```$ conda activate gensvs_eval_env```
2. Install additional python dependencies: ```$ pip install -r ./env_info/svs_eval_env.txt```
3. Build the [ViSQOL API](https://github.com/google/visqol) and place in folder within the root directory
   - >**Note:** URL and SHA256 of Armadillo headers in the file "WORKSPACE" need to be changed to a recent version (https://sourceforge.net/projects/arma/files/)
4. To compute metrics with the PEASS-toolkit in the folder [```./01_evaluation_and_correlation/peass_v2.0.1```](https://github.com/pablebe/gensvs_eval/tree/main/01_evaluation_and_correlation/peass_v2.0.1) Matlab (version<2025a) is required to execute Python's Matlab engine   

### III. ```gensvs_fad_mse_eval_env```: Conda environment for evaluation intrusive embedding-based metrics
For computation of FAD and MSE evaluation metrics in addition to MERT/Music2Latent MSEs you'll need an additional conda environment. Please follow the steps below to set the environment up:
1. Create the conda environment: ```$ conda env create -f ./env_info/svs_fad_mse_eval_env.yml``` 
2. Install additional python dependencies: ```$ pip install -r ./env_info/svs_fad_mse_eval_env.txt```  
3. Test fadtk installation with: ```$ python -m fadtk.test```

## 🧮 Evaluation and Correlation Analysis

Within the folder [```01_evaluation_and_correlation```](https://github.com/pablebe/gensvs_eval/tree/main/01_evaluation_and_correlation) all code to compute all objective evaluation metrics, the evaluation of the DCR test results and the correlation analysis of the paper are collected.

All metrics, listening test results, DMOS ratings, and the audio used for both metric computation and loudness-normalized listening tests are available on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15911723.svg)](https://doi.org/10.5281/zenodo.15911723), where a single CSV file combines all computed metrics, DMOS ratings, and individual ratings for each audio sample. The dataset also includes instructions and an example Python script to help you benchmark your own audio quality metric by calculating correlation coefficients on our dataset. If you are only interested in the final data, you can ignore the individual CSV files located in the [```./04_evaluation_data```](https://github.com/pablebe/gensvs_eval/tree/main/03_evaluation_data) folder, as they reflect intermediate steps during the paper’s development.

### Reproduce the evaluation carried out in the paper 
If you want to reproduce how we calculated all metrics, please download the data from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15911723.svg)](https://doi.org/10.5281/zenodo.15911723) and copy the data into the root directory. Then follow the steps below: 

### Compute Objective Evaluation Metrics
To calculate all objective metrics mentioned in the paper three python scripts are necessary. The evaluation of PAM as well as the FAD & MSE metrics are carried out in separate scripts. For the computation of the FAD & MSE metrics the conda environment ```gensvs_fad_mse_eval_env``` is necessary. All other metrics can be computed with the ```gensvs_eval_env```.

#### Compute FAD & MSE metrics 
To compute the FAD and MSE metrics we modified the code of Microsoft's fadtk [5]. The modified code can be found in ```./01_evaluation_and_correlation/fadtk_mod```.
The metrics can be computed with a python script. To show how the evaluation script is called we have added a examplary bash scripts.
- Python: [```./01_evaluation_and_correlation/gensvs_eval_fad_mse.py```](https://github.com/pablebe/gensvs_eval/blob/main/01_evaluation_and_correlation/gensvs_eval_fad_mse.py)
- Bash: [```$ ./01_evaluation_and_correlation/gensvs_eval_fad_mse.sh```](https://github.com/pablebe/gensvs_eval/blob/main/01_evaluation_and_correlation/gensvs_eval_fad_mse.sh)
- Required Conda environments:
   - II. ```gensvs_fad_mse_eval_env```
#### Compute PAM scores
To compute the PAM scores of https://github.com/soham97/PAM please use the following scripts:
- Python: [```./01_evaluation_and_correlation/gensvs_eval_pam.py```](https://github.com/pablebe/gensvs_eval/blob/main/01_evaluation_and_correlation/gensvs_eval_pam.py)
- Bash: [```$ 01_evaluation_and_correlation/gensvs_eval_pam.sh```](https://github.com/pablebe/gensvs_eval/blob/main/01_evaluation_and_correlation/gensvs_eval_pam.sh)
- Required Conda environments:
   - I. ```gensvs_eval_env```
#### Compute all other non-intrusive and intrusive metrics (BSS-Eval, PEASS, SINGMOS, XLS-R-SQA, Audiobox-AES)
To compute all other metrics mentioned in the paper please use
- Python: [```./01_evaluation_and_correlation/gensvs_eval_metrics.py```](https://github.com/pablebe/gensvs_eval/blob/main/01_evaluation_and_correlation/gensvs_eval_metrics.py)
- Bash: [```$ ./01_evaluation_and_correlation/gensvs_eval_metrics.sh```](https://github.com/pablebe/gensvs_eval/blob/main/01_evaluation_and_correlation/gensvs_eval_metrics.py)
- Required Conda environments:
   - I. ```gensvs_eval_env```
### Subjective Evaluation and correlation analysis
In order to evaluate the DMOS data, reproduce the correlation analysis results and export all the plots and *.csv files, the Python script can be executed under:
- Python: [```./01_evaluation_and_correlation/gensvs_eval_plots_and_csv_export.py```](https://github.com/pablebe/gensvs_eval/blob/main/01_evaluation_and_correlation/gensvs_eval_plots_and_csv_export.py)
- Required Conda environments:
   - I. ```gensvs_eval_env```

## 🏋🏽‍♀️🏃🏽‍♀️‍➡️ Training and Inference
>**Note:** The inference of the generative models (SGMSVS and MelRoFo (S) + BigVGAN) are executable using [gensvs](https://pypi.org/project/gensvs/). The inference can be carried out using either the included command line tool or a Python script (see [gensvs](https://pypi.org/project/gensvs/)).

The folder [```00_training_and_inference```](https://github.com/pablebe/gensvs_eval/tree/main/00_training_and_inference) contains all code required to carry out training and inference for all the models mentioned in the paper.
To carry out this code please set up the conda environment ```gensvs_env``` with [gensvs](https://pypi.org/project/gensvs/) installed (see I. above)

### Trained models
 - HTDemucs, MelRoFo (S), SGMSVS and the finetuned BigVGAN checkpoints (generator and discriminator) can be downloaded from [Hugging Face](https://huggingface.co/collections/pablebe/gensvs-eval-model-checkpoints-687e1c967b43f867f34d6225)
 - The model checkpoint for MelRoFo (L) can be downloaded [here](https://huggingface.co/KimberleyJSN/melbandroformer/blob/main/MelBandRoformer.ckpt).

### Inference
If you want to run the inference of all models using the Python code provided in this GitHub repository you can either use the python inference scripts directly or use the provided bash scripts which show how to call the python scripts:
#### Discriminative baseline models: HTDemucs, MelRoFo (S) & MelRoFo (L)
To run the mask-based baseline models on a folder of musical mixtures the following scripts can be used:  
- Python:
  - HTDemucs & MelRoFo (S): [```00_training_and_inference/inference_baseline.py```](https://github.com/pablebe/gensvs_eval/blob/main/00_training_and_inference/inference_baseline.py)
  - MelRoFo (L): [```00_training_and_inference/inference_melroformer_large.py```](https://github.com/pablebe/gensvs_eval/blob/main/00_training_and_inference/infer_melroformer_large.sh)
- Bash:
  - HTDemucs: [```$ 00_training_and_inference/infer_htdemucs.sh```](00_training_and_inference/infer_htdemucs.sh)
  - MelRoFo (S): [```$ 00_training_and_inference/infer_melroformer_small.sh```](https://github.com/pablebe/gensvs_eval/blob/main/00_training_and_inference/infer_melroformer_small.sh)
  - MelRoFo (L): [```$ 00_training_and_inference/infer_melroformer_large.sh```](https://github.com/pablebe/gensvs_eval/blob/main/00_training_and_inference/infer_melroformer_large.sh)
 <!-- - III. ```gensvs_eval_baseline_env``` -->

#### Generative models: SGMSVS & MelRoFo (S) + BigVGAN
*pip-package to infere generative models on new data coming soon*! 
<!-- 
To run the generative models on a folder of musical mixtures the following scripts can be used:  
- Python: 
  - SGMSVS: ```00_training_and_inference/inference_sgmsvs.py```
  - MelRoFo (S) + BigVGAN: ```00_training_and_inference/inference_melroformer_small_bigvgan.py```
- Bash:
  - SGMSVS: ```$ 00_training_and_inference/infer_sgmsvs.sh```
  - MelRoFo (S) + BigVGAN: ```00_training_and_inference/infer_melroformer_small_bigvgan.sh```
- Required Conda environment:
   - SGMSVS: IV. ```gensvs_eval_sgmsvs_env```
   - MelRoFo (S) + BigVGAN: V. ```gensvs_eval_bigvgan_env``` -->
### Training
Below the python training scripts and example bash scripts are listed for all models to reproduce all trainings and the set parameters mentioned in the paper:
#### Discriminative Baselines: HTDemucs & MelRoFo (S)
To train the dicriminative mask-based baselines use:
  - Python:
    - HTDemucs & MelRoFo (S): [```00_training_and_inference/train_baseline.py```](https://github.com/pablebe/gensvs_eval/blob/main/00_training_and_inference/train_baseline.py)
  - Bash: 
    - HTDemucs: [```$ 00_training_and_inference/train_htdemucs.sh```](https://github.com/pablebe/gensvs_eval/blob/main/00_training_and_inference/train_htdemucs.sh)
    - MelRoFo (S): [```$ 00_training_and_inference/train_melroformer.sh```](https://github.com/pablebe/gensvs_eval/blob/main/00_training_and_inference/train_melroformer.sh)
#### Generative models: SGMSVS & MelRoFo (S) + BigVGAN
To train the SGMSVS model or task-specifically finetune BigVGAN for singing voice separation with MelRoFo (S) you can use:

- Python: 
   - SGMSVS: [```00_training_and_inference/train_sgmsvs.py```](https://github.com/pablebe/gensvs_eval/blob/main/00_training_and_inference/train_sgmsvs.py)
   - MelRoFo (S) + BigVGAN: [```00_training_and_inference/train_finetune_bigvgan.py```](https://github.com/pablebe/gensvs_eval/blob/main/00_training_and_inference/train_finetune_bigvgan.py)
- Bash: 
   - SGMSVS: [```$ 00_training_and_inference/train_sgmsvs.sh```](https://github.com/pablebe/gensvs_eval/blob/main/00_training_and_inference/train_sgmsvs.sh)
   - MelRoFo (S) + BigVGAN: [```$ 00_training_and_inference/train_bigvgan.sh```](https://github.com/pablebe/gensvs_eval/blob/main/00_training_and_inference/train_bigvgan.sh)
   <!-- - SGMSVS: II. ```gensvs_eval_sgmsvs_env```
   - MelRoFo (S) + BigVGAN: III. ```gensvs_eval_bigvgan_env``` -->

## Third-party code
All third-party code is contained in separate folders, each of which is specifically listed in this README.md file, if there exist LICENSE files for these third party folders they are located within their respective directories.

The third-party directories are:

- [```./00_training_and_inference/sgmsvs/sgmse```](https://github.com/pablebe/gensvs_eval/tree/main/00_training_and_inference/sgmsvs/sgmse) (from https://github.com/sp-uhh/sgmse)
- [```./01_evaluation_and_correlation/pam_eval```](https://github.com/pablebe/gensvs_eval/tree/main/01_evaluation_and_correlation/pam_eval) (from https://github.com/soham97/PAM)
- [```./01_evaluation_and_correlation/peass_v2.0.1```](https://github.com/pablebe/gensvs_eval/tree/main/01_evaluation_and_correlation/peass_v2.0.1) (from https://gitlab.inria.fr/bass-db/peass/-/tree/master/v2.0.1)
- [```./00_training_and_inference/bigvgan_utils```](https://github.com/pablebe/gensvs_eval/tree/main/00_training_and_inference/bigvgan_utils) (from https://github.com/NVIDIA/BigVGAN)
- [```./00_training_and_inference/baseline_models/backbones```](https://github.com/pablebe/gensvs_eval/tree/main/00_training_and_inference/baseline_models/backbones) (from https://github.com/ZFTurbo/Music-Source-Separation-Training and https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)

## References and Acknowledgements
The inference code for the SGMSVS model was built upon the code made available in:
```bib
@article{richter2023speech,
         title={Speech Enhancement and Dereverberation with Diffusion-based Generative Models},
         author={Richter, Julius and Welker, Simon and Lemercier, Jean-Marie and Lay, Bunlong and Gerkmann, Timo},
         journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
         volume={31},
         pages={2351-2364},
         year={2023},
         doi={10.1109/TASLP.2023.3285241}
        }
```
The inference code for MelRoFo (S) + BigVGAN was put together from the code available at:
```bib
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks}, 
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      howpublished={\url{https://github.com/ZFTurbo/Music-Source-Separation-Training}},
      primaryClass={cs.SD},
      url={https://github.com/ZFTurbo/Music-Source-Separation-Training}
      }
```
```bib
@misc{jensen2024melbandroformer,
      author       = {Kimberley Jensen},
      title        = {Mel-Band-Roformer-Vocal-Model},
      year         = {2024},
      howpublished = {\url{https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model}},
      note         = {GitHub repository},
      url          = {https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model}
    }
```
```bib
@inproceedings{lee2023bigvgan,
               title={BigVGAN: A Universal Neural Vocoder with Large-Scale Training},
               author={Sang-gil Lee and Wei Ping and Boris Ginsburg and Bryan Catanzaro and Sungroh Yoon},
               booktitle={in Proc. ICLR, 2023},
               year={2023},
               url={https://openreview.net/forum?id=iTtGCMDEzS_}
              }
```
The whole evaluation code was created using Microsoft's [Frechet Audio Distance Tookit](https://github.com/microsoft/fadtk/tree/main) as a template
```bib
@inproceedings{fadtk,
               title = {Adapting Frechet Audio Distance for Generative Music Evaluation},
               author = {Azalea Gui, Hannes Gamper, Sebastian Braun, Dimitra Emmanouilidou},
               booktitle = {Proc. IEEE ICASSP 2024},
               year = {2024},
               url = {https://arxiv.org/abs/2311.01616},
              }
```
If you use the [MERT](https://huggingface.co/m-a-p/MERT-v1-95M) or [Music2Latent](https://github.com/SonyCSLParis/music2latent) MSE please also cite the initial work in which the embeddings were proposed. 

For [MERT](https://huggingface.co/m-a-p/MERT-v1-95M):
```bib
@misc{li2023mert,
      title={MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training}, 
      author={Yizhi Li and Ruibin Yuan and Ge Zhang and Yinghao Ma and Xingran Chen and Hanzhi Yin and Chenghua Lin and Anton Ragni and Emmanouil Benetos and Norbert Gyenge and Roger Dannenberg and Ruibo Liu and Wenhu Chen and Gus Xia and Yemin Shi and Wenhao Huang and Yike Guo and Jie Fu},
      year={2023},
      eprint={2306.00107},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
For [Music2Latent](https://github.com/SonyCSLParis/music2latent):
```bib
@inproceedings{pasini2024music2latent,
  author       = {Marco Pasini and Stefan Lattner and George Fazekas},
  title        = {{Music2Latent}: Consistency Autoencoders for Latent Audio Compression},
  booktitle    = ismir,
  year         = 2024,
  pages        = {111-119},
  venue        = {San Francisco, California, USA and Online},
  doi          = {10.5281/zenodo.14877289},
}
```

## Citation
If you use any parts of our code, our data or the gensvs package in your work, please cite our paper and the work that formed the basis of this research.

Our paper can be cited with:
```bib
@INPROCEEDINGS{bereuter2025,
  author={Bereuter, Paul A. and Stahl, Benjamin and Plumbley, Mark D. and Sontacchi, Alois},
  booktitle={2025 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)}, 
  title={Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation Models}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Measurement;Degradation;Training;Time-frequency analysis;Correlation;Limiting;Computational modeling;Conferences;Reliability;Software development management},
  doi={10.1109/WASPAA66052.2025.11230934}
  }
```
