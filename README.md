# CodingLab: Deep Reinforcement Learning

## Introduction

This repository is the final course release combining the work of the three team members:

- **Lejla Mehmedović:** PPO algorithms
- **Fang Ning Hsu:** Standard DQN algorithms
- **Lirong Zhuang:** Rainbow DQN algorithms


## Final results

The selected model for each variant outperformed the greedy baseline on the final test set:

| Variant | Selected model | Final Test reward | Greedy reward |
|:---|:---|:---|:---|
| 0 | Rainbow DQN | 222.72 | 216.920 |
| 1 | Rainbow DQN | 399.44 | 384.255 |
| 2 | PPO | 261.73 | 249.750 |

## Project structure

```text
.
|-- main.py
|-- greedy.py
|-- test_policy.py
|-- requirements.txt
|-- algorithms/
|   |-- dqn.py
|   |-- rainbow.py
|   |-- pretrained_model.py
|   `-- ppo.py
|-- environments/
|   |-- environment.py
|   |-- dqn_environment.py
|   |-- ppo_environment.py
|   `-- rainbow_environment.py
|-- scripts/
|   |-- autoencoder/
|   |   |-- autoencoder_model.py
|   |   `-- pretrain_autoencoder.py
|   `-- dqn/
|       `-- spatial_distribution.py
`-- data/
    |-- variant_0/
    |-- variant_1/
    `-- variant_2/
```


## Create the virtual environment

Python 3.11 is recommended.

### Windows PowerShell

```powershell
python -m venv codinglab_drl
.\codinglab_drl\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv codinglab_drl
source codinglab_drl/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The generated `codinglab_drl/` directory is ignored by Git. It contains local dependencies only and must not be committed.

### Optional: CUDA-enabled PyTorch for GPU training

If GPU training is required, replace the default PyTorch installation with the CUDA 12.8 build:

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Train

All training commands must be run while the `codinglab_drl` virtual environment is activated. If a new terminal is opened, activate it again before training:

```powershell
# Windows PowerShell
.\codinglab_drl\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source codinglab_drl/bin/activate
```

After activation, all algorithms use `main.py` and the existing episode data under `data/variant_<n>`.

```powershell
# DQN training example
python main.py --algorithm dqn --variant 0 --num_episodes 10000

# Rainbow DQN training example
python main.py --algorithm rainbow --variant 0 --num_episodes 10000

# PPO training example
python main.py --algorithm ppo --variant 0 --num_episodes 10000
```

Variants `0`, `1` and `2` are supported. Change `--model_version` when repeating a run with the same configuration. Checkpoints are named `dqn_vX_variant_N.pt`, `rainbow_dqn_vX_variant_N.pt` or `ppo_vX_variant_N.pt` under `outputs/checkpoints/`; matching TensorBoard directories are created under `outputs/tensorboard/`.

The latest PPO defaults to the `hybrid` network. Use `--ppo_architecture cnn` or `--ppo_architecture mlp` for the two final ablations. Behavioural cloning defaults to 2,000 demonstration episodes and 20 epochs; `--bc_episodes` and `--bc_epochs` can override these values. For a quick installation smoke test, use:

```powershell
python main.py --algorithm ppo --variant 0 --num_episodes 1 --bc_episodes 0
```


## Test

`test_policy.py` selects the same environment and model class as `main.py`. DQN and Rainbow checkpoints contain Q-network weights; PPO checkpoints also record their architecture. Use `--ppo_architecture` only when testing a raw PPO state dictionary without architecture metadata.

```powershell
python test_policy.py --algorithm dqn --variant 0 --model_path PATH_TO_DQN_MODEL.pt
python test_policy.py --algorithm rainbow --variant 0 --model_path PATH_TO_RAINBOW_MODEL.pt
python test_policy.py --algorithm ppo --variant 0 --model_path PATH_TO_PPO_MODEL.pt
```

Each command evaluates 100 test episodes by default and prints the average reward.