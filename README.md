# CodingLab_DRL_Zhuang



# Setup

## Python version

Python 3.11 is recommended. This project was tested with Python 3.11.6.

## Create virtual environment

python -m venv .venv

## Activate

Linux/macOS:

source .venv/bin/activate

Windows:

.\.venv\Scripts\Activate.ps1

## Install dependencies

pip install -r requirements.txt

+

python -m pip install torch torchvision torchaudio

or

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


## Run

python main.py

## Tensor-Board

tensorboard --logdir logs
