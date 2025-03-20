Help me with the training.

My laptop only has 8 GB of RAM. It will take forever.

How to train:

You will run by command:

for example:

python train.py --batch_size 8 --num_epochs 20 --dataset_path './dataset' --save_model './model_epoch.pth' --num_workers 8


quite similar for evaluation


For the data set. I use the HR version on Kaggle.

You can use my downscale.py to generate the LR version.


Good luck.

Lam Nguyen


Safa:

Mildy-professional README below. In progress

# Project Name Here

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

Safa: I ran this lol

`python train.py --batch_size 16 --num_epochs 100 --dataset_path dataset --save_model './safa_100_epoch_results/model_epoch.pth' --num_workers 8`
