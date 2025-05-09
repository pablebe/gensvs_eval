# Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation
This repository accompanies the submission titled "Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation".

The repository is organized in four subfolders with the following contents:

## 00_training_and_inference
Contains all code required to carry out training and inference for all the models mentionedin the paper.

### ./00_training_and_inference/train_baseline.py
This python-script was adapted from [1] and can be used to train HTDemucs and Mel-RoFo. (S)

### ./00_training_and_inference/train_sgmsvs.py
This python-script was adapted from [1] and can be used to train the SGMSVS model (train_sgmsvs.py).

### ./00_training_and_inference/train_finetuned_bigvgan.py
This python-script was adapted from [2] and can be used to task-specifically finetune the vocoder-based BigVGAN for singing voice separation.

## 01_objective_evaluation
Contains the code to compute all objective evaluation metrics and all *.csv files of the evaluated objective audio quality metrics.

Note: to compute ViSQOL metrics the ViSQOL API from [3] has to be installed locally. 

### Training
### Inference

## 02_subjective_evaluation
This folder contains the DMOS data collected with a degredation category rating. \n
It also contains code to analyze the listening test results. 

## 03_correlation_analysis

## 04_audio_examples
12 audio examples of each model can be found in this folder.

## env_info
Contains the *.yml files to create the conda environments and *.txt requirements to install all packages using pip.
All environments were tested using a Debian machine. 
To create one of the environments carry out the following steps:
1.) conda create 
2.) pip install ...

## trained models
The trained models used in this paper can be downloaded from <add-url-here>

## Third party code
### Third party folders
All third-party code is contained in separate folders, each of which is specifically listed in this README.md file, and the corresponding LICENSE files for each folder are located within their respective directories.

The used third party folders are the following:
- Path: 00_training_and_inference/sgmsvs/sgmse
    - Reference: [1]

## References:
[1]
