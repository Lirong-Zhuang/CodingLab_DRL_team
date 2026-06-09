# CodingLab_DRL_Zhuang



# Setup

## Python version

Python 3.11 is recommended. This project was tested with Python 3.11.6.

## Create virtual environment

```
python -m venv .venv
```

## Activate

Linux/macOS:

```
source .venv/bin/activate
```

Windows:

```
.\.venv\Scripts\Activate.ps1
```

## Install dependencies

```
pip install -r requirements.txt
```

and Pytorch, choose the version with or without CUDA according to your device

```
python -m pip install torch torchvision torchaudio
```

or

```
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Run

```
python main.py
```

## Tensor-Board

```
tensorboard --logdir logs
```


# Train in Cluster

## Upload Files

```
cd ..../CodingLab_DRL_team
sftp <Account>@<host>

cd code_for_zhuang

put ....
put -r ...

bye
```

## SSH Connection

```
ssh <Account>@<host>

cd code_for_zhuang
```

## Create Enroot Environment

```
cd ~/code_for_zhuang

salloc -p lrz-dgx-a100-80x8 --gres=gpu:1
srun --pty bash

enroot import docker://nvcr.io#nvidia/pytorch:24.12-py3

ls *.sqsh

enroot create --name codinglab_drl_zhuang <IMAGE_NAME>.sqsh
enroot start --root --rw codinglab_drl_zhuang

pip install pandas scipy tensorboard==2.20.0 tensorflow==2.20.0

exit

enroot export -o tf_container_dgx-a100-80.sqsh codinglab_drl_zhuang
ls -lh tf_container_dgx-a100-80.sqsh
```

## Check Mission Process

```
squeue -u $USER
```

## Submit Mission

```
bash cluster/master_GPU.sh 
```

## Download Files

```
cd /home/zhuanglr/Documents/TUM/CodingLab/CodingLab_DRL_team

sftp <Account>@<host>

cd code_for_zhuang

get <File Name>.out

get models/<Model Name>.pt

get -r logs2/DQN_v8.5.13_variant_0

bye
```
