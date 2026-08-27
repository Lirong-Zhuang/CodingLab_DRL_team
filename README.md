# CodingLab Deep Reinforcement Learning

This repository is the final course release combining the work of the three team members:

- **Lejla Mehmedović:** PPO with behavioural-cloning warm start.
- **Fang Ning Hsu:** standard and CNN-based DQN implementations.
- **Lirong Zhuang:** Rainbow DQN and autoencoder experiments.

The release intentionally keeps the original `main.py` training structure. Experiment logs, trained models, intermediate statistics and visualizations are not included.

The release sources are pinned to the latest team branches fetched on 27 August 2026:

| Component | Team branch | Source revision | Release default |
|:---|:---|:---|:---|
| Course scaffold | `main` | `bdbfba73` | original environment, greedy baseline and test template |
| Lirong Rainbow | `dev_zhuang` | `e7c7face` | final Rainbow CNN |
| Lejla PPO | `lejla` | `ad660fd6` | hybrid actor-critic with BC warm start |
| Ashley DQN | `dev_ashley_new` | `e7a73956` | CNN-v2; action-masked CNN-v2 for Variant 2 |

The `environments/environment.py` and `greedy.py` files are retained directly from `main`, with only the import updated after moving the environment into its package. The original `test_policy.py` loop is completed with the three release environments, policy construction and checkpoint loading. The DQN and PPO implementations are copied directly from the listed team revisions. Their release changes are limited to package-relative import paths, the shared `main.py` integration, and Ashley's `eisode_steps` typo, which otherwise prevents CNN-v2 from starting. With the Rainbow author's authorization, `algorithms/rainbow.py` keeps only the original v8 agent and its required classes, and renames the agent class to `rainbow_dqn` without changing its hyperparameters or training logic.

## Final results

The selected model for each variant outperformed the greedy baseline on the final test set:

| Variant | Selected model | Test reward | Greedy reward |
|:---|:---|:---|:---|
| 0 | Rainbow DQN | 222.72 | 216.920 |
| 1 | DQN | 395.91 | 384.255 |
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

The Rainbow `Environment` is derived from the final environment on the Rainbow development line. It produces a `7 x 5 x 5` CNN observation containing agent position, load, remaining time, item presence, item lifetime, reachability and heuristic value.

Each algorithm uses its corresponding environment automatically:

- Ashley's `Environment` provides the final seven-channel CNN observation. Variant 2 automatically uses the final legal-action-masked CNN-v2 agent.
- Lejla's `Environment` provides the engineered state features. The latest hybrid actor-critic is the default; `cnn` and `mlp` ablations are also available.
- Lirong's Rainbow `Environment` provides the final seven-channel observations used by Rainbow.

The shared entry point uses one Rainbow agent:

- `rainbow_dqn`: the original `DQN_v8` end-to-end Rainbow CNN implementation under its release name.

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
# Ashley CNN-DQN (CNN-v2; Variant 2 enables action masking automatically)
python main.py --algorithm dqn --variant 0 --num_episodes 10000

# Final Rainbow DQN
python main.py --algorithm rainbow --variant 0 --num_episodes 10000

# Lejla PPO
python main.py --algorithm ppo --variant 0 --num_episodes 10000
```

Variants `0`, `1` and `2` are supported. Change `--model_version` when repeating a run with the same configuration. Checkpoints are named `dqn_vX_variant_N.pt`, `rainbow_dqn_vX_variant_N.pt` or `ppo_vX_variant_N.pt` under `outputs/checkpoints/`; matching TensorBoard directories are created under `outputs/tensorboard/`.

Lejla's latest PPO defaults to the `hybrid` network. Use `--ppo_architecture cnn` or `--ppo_architecture mlp` for the two final ablations. Behavioural cloning defaults to 2,000 demonstration episodes and 20 epochs; `--bc_episodes` and `--bc_epochs` can override these values. For a quick installation smoke test, use:

```powershell
python main.py --algorithm ppo --variant 0 --num_episodes 1 --bc_episodes 0
```

To verify every training entry point after installation, run:

```powershell
python main.py --algorithm dqn --variant 0 --num_episodes 1 --model_version 0
python main.py --algorithm dqn --variant 1 --num_episodes 1 --model_version 0
python main.py --algorithm dqn --variant 2 --num_episodes 1 --model_version 0
python main.py --algorithm rainbow --variant 0 --num_episodes 1 --model_version 0
python main.py --algorithm rainbow --variant 1 --num_episodes 1 --model_version 0
python main.py --algorithm rainbow --variant 2 --num_episodes 1 --model_version 0
python main.py --algorithm ppo --variant 0 --num_episodes 1 --bc_episodes 0 --model_version 0
python main.py --algorithm ppo --variant 1 --num_episodes 1 --bc_episodes 0 --model_version 0
python main.py --algorithm ppo --variant 2 --num_episodes 1 --bc_episodes 0 --model_version 0
```

These one-episode commands only verify that training starts; a checkpoint is not expected before the first validation phase. Use a different `--model_version` when repeating them because each run creates a TensorBoard directory under `outputs/`.

## Test

`test_policy.py` selects the same environment and model class as `main.py`. DQN and Rainbow checkpoints contain Q-network weights; PPO checkpoints also record their architecture. Use `--ppo_architecture` only when testing a raw PPO state dictionary without architecture metadata.

```powershell
python test_policy.py --algorithm dqn --variant 0 --model_path PATH_TO_DQN_MODEL.pt
python test_policy.py --algorithm rainbow --variant 0 --model_path PATH_TO_RAINBOW_MODEL.pt
python test_policy.py --algorithm ppo --variant 0 --model_path PATH_TO_PPO_MODEL.pt
```

Each command evaluates 100 test episodes by default and prints the average reward. Use `--num_test_episodes 1` for a quick invocation check. The original greedy baseline remains available with `python greedy.py`.
