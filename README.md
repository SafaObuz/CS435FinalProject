
# Implementation of Dual Aggregation Transformer for Super-Resolution of Low Resolution Video

## Installation

### Package Configuration

Run `setup.ps1` if on Windows or `setup.sh` if on Linux (Note Mac is not supported as there is no CUDA support)

If in VSCode, ***press cmd + shift + P***, search ***"Python: Select Interpreter"*** and select the Python interpreter that belongs to your venv folder.
- Linux: ***CS435FinalProject/venv/bin/python***
- Windows: ***CS435FinalProject\venv\Scripts\python.exe***

### Dataset

Download and unzip the *DIV2K* dataset from [kaggle](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images/data). Place the corresponding training and validation datasets into `./dataset/DIV2K_train_HR` and `./dataset/DIV2K_valid_HR` respectively. 

Ensure not to nest the folders. Feel free to remove the folder and then copy the entirety of the unzipped contents into `./dataset`

Then, run 

- `python downscale.py ./dataset/DIV2K_train_HR ./dataset/DIV2K_train_LR`

- `python downscale.py ./dataset/DIV2K_valid_HR ./dataset/DIV2K_valid_LR` 

to generate the low-resolution training and validation set. 

## Training

Example run command

`python train.py --batch_size 16 --num_epochs 100 --dataset_path dataset --save_model './safa_100_epoch_results/model_epoch.pth' --num_workers 8`
