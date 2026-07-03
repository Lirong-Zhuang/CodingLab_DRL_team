import argparse
import time
from torch.utils.tensorboard import SummaryWriter
import dqn
import ppo
import rainbow_dqn


# Edit these values before running this file directly.
VARIANT = 0
SEED = 777
DATA_DIR = './data'
NETWORK_VERSION = 8
ENV_VERSION = 11
MODEL_VERSION = 23
NUM_EPISODES = 30000
ENCODER_PATH = None
FREEZE_ENCODER = False
ENCODER_TYPE = "cnn"
GB_WINDOW = 0
GB_WINDOWS = [1000, 2000, 3000, 4000, 5000]


class Config:
    variant = VARIANT
    seed = SEED
    data_dir = DATA_DIR
    env_version = ENV_VERSION
    network_version = NETWORK_VERSION
    model_version = MODEL_VERSION
    num_episodes = NUM_EPISODES
    encoder_path = ENCODER_PATH
    freeze_encoder = FREEZE_ENCODER
    encoder_type = ENCODER_TYPE
    gb_window = GB_WINDOW


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, default=VARIANT, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--network_version", type=int, default=NETWORK_VERSION)
    parser.add_argument("--env_version", type=int, default=ENV_VERSION)
    parser.add_argument("--model_version", type=int, default=MODEL_VERSION)
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--encoder_path", type=str, default=ENCODER_PATH)
    parser.add_argument("--freeze_encoder", action="store_true", default=FREEZE_ENCODER)
    parser.add_argument("--encoder_type", type=str, default=ENCODER_TYPE, choices=["cnn", "lenet5", "alexnet8", "resnet18"])
    parser.add_argument("--gb_window", type=int, default=GB_WINDOW)
    return parser.parse_args()


args = parse_args()


# set seed
seed = args.seed  # TODO: set seed to allow for reproducibility of results

import os
os.environ['PYTHONHASHSEED'] = str(seed)

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)

import torch
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# initialize environment
from environment import Environment
from environment_v2 import Environment_v2
from environment_v3 import Environment_v3
from environment_v4 import Environment_v4
from environment_v5 import Environment_v5
from environment_v6 import Environment_v6
from environment_v7 import Environment_v7
from environment_v8 import Environment_v8
from environment_v9 import Environment_v9
from environment_v10 import Environment_v10
from environment_v11 import Environment_v11
from environment_v12 import Environment_v12

data_dir = args.data_dir  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = args.variant  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
model_dir = './models2'
os.makedirs(model_dir, exist_ok=True)
os.makedirs('./logs2', exist_ok=True)


def build_env(env_version, variant, data_dir):
    env_classes = {
        1: Environment,
        2: Environment_v2,
        3: Environment_v3,
        4: Environment_v4,
        5: Environment_v5,
        6: Environment_v6,
        7: Environment_v7,
        8: Environment_v8,
        9: Environment_v9,
        10: Environment_v10,
        11: Environment_v11,
        12: Environment_v12,
    }
    return env_classes[env_version](variant, data_dir)


def build_dqn_network(network_version, env):
    network_classes = {
        5: dqn.DQN_v5,
        6: rainbow_dqn.DQN_v6,
        7: rainbow_dqn.DQN_v7,
        8: rainbow_dqn.DQN_v8,
        9: rainbow_dqn.DQN_v9,
        10: rainbow_dqn.DQN_v10,
        11: rainbow_dqn.DQN_v11,
    }
    if network_version == 11:
        encoder_path = args.encoder_path
        if isinstance(encoder_path, str) and encoder_path.lower() in {"", "none", "null"}:
            encoder_path = None
        return rainbow_dqn.DQN_v11(
            env,
            encoder_path=encoder_path,
            freeze_encoder=args.freeze_encoder,
            encoder_type=args.encoder_type,
        )
    return network_classes[network_version](env)


env = build_env(args.env_version, variant, data_dir)


# DQN validation function
def validate_dqn(env, network, num_validation_episodes):
    old_epsilon = network.epsilon  # save current epsilon value
    network.epsilon = 0  # set epsilon to 0 for validation

    was_training = network.q_network.training
    network.q_network.eval()

    vali_rew = 0
    num_episodes = num_validation_episodes

    for episode in range(num_episodes):
        obs = env.reset('validation')
        episode_reward = 0

        for step in range(env.episode_steps):
            act = network.select_action(obs)
            rew, next_obs, done = env.step(act)
            obs = next_obs
            episode_reward += rew

            if done:
                break

        vali_rew += episode_reward


    network.epsilon = old_epsilon

    if was_training:
        network.q_network.train()

    avg_reward = vali_rew / num_episodes
    return avg_reward

# DQN training function
def train_dqn(env):

    ##--------Version Information--------##
    network = build_dqn_network(args.network_version, env)
    if args.num_episodes is not None:
        network.num_episodes = args.num_episodes
        if hasattr(network, 'epsilon_decay_steps'):
            network.epsilon_decay_steps = network.num_episodes * 0.8
    model_version = args.model_version
    note = (
        f'network_version: {args.network_version}, env_version: {args.env_version}, '
        f'encoder_type: {args.encoder_type}, encoder_path: {args.encoder_path}, '
        f'freeze_encoder: {args.freeze_encoder}'
    )
    ##----------------------------------##

    start_time = time.time()
    train_rew = 0  # initialize reward tracking
    num_episodes = network.num_episodes
    output_interval = 20

    # best model tracking
    model_path = os.path.join(model_dir, f'{network.file_name}{env.env_name}{model_version}_variant_{env.variant}.pt')
    best_vali_rew = -float('inf')
    gb_windows = sorted(
        {
            window
            for window in GB_WINDOWS + ([args.gb_window] if args.gb_window else [])
            if window > 0
        }
    )
    gb_model_paths = {
        window: os.path.join(
            model_dir,
            f'{network.file_name}{env.env_name}{model_version}_variant_{env.variant}_GB{window}.pt'
        )
        for window in gb_windows
    }
    best_gb_rews = {window: -float('inf') for window in gb_windows}
    gb_start_episodes = {
        window: max(1, num_episodes - window + 1)
        for window in gb_windows
    }

    # print info
    print(f'Training {network.network_name} on Variant {env.variant}')

    # validation info
    vali_interval = 100
    num_validation_episodes = 100 # validation episodes in each validation phase

    # file protection
    run_name = f'{network.file_name}{env.env_name}{model_version}_variant_{env.variant}'
    log_dir = f'./logs2/{run_name}'
    existing_gb_model_paths = [
        path
        for path in gb_model_paths.values()
        if os.path.exists(path)
    ]
    if os.path.exists(model_path) or existing_gb_model_paths or os.path.exists(log_dir):
        raise FileExistsError(
            f'Model name already exists: {run_name}. '
            f'Please increase model_version before training.'
        )

    # tensor-board writer
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Hyperparameters', f'num_episodes: {network.num_episodes}, batch_size: {network.batch_size}, gamma: {network.gamma}, learning_rate: {network.learning_rate}, epsilon_start: {network.epsilon_start}, epsilon_end: {network.epsilon_end}, epsilon_decay_steps: {network.epsilon_decay_steps}, target_update_freq: {network.target_update_freq}', 0)
    writer.add_text('Model_info', f'Model: {run_name}, Variant: {env.variant}', 0)
    writer.add_text('Note', note, 0)
    writer.add_text('GB', f'gb_windows: {gb_windows}, gb_model_paths: {gb_model_paths}', 0)

    # running
    for episode in range(num_episodes):
        obs = env.reset('training')
        episode_reward = 0
        episode_losses = []

        for step in range(env.episode_steps):
            act = network.select_action(obs)
            rew, next_obs, done = env.step(act)

            # let the model store the transition if needed
            network.store_transition(obs, act, rew, next_obs, done)

            obs = next_obs
            episode_reward += rew

            # update the model when it is ready
            if network.ready_to_update():
                loss = network.optimize_model()
                episode_losses.append(loss)
            if done:
                break

        # run algorithm-specific end-of-episode updates
        network.update_after_episode(episode)

        # print training progress
        if (episode + 1) % output_interval == 0:

            elapsed_time = time.time() - start_time
            episodes_done = episode + 1
            avg_time_per_episode = elapsed_time / episodes_done
            remaining_episodes = num_episodes - episodes_done
            estimated_remaining_time = avg_time_per_episode * remaining_episodes

            remaining_hours = estimated_remaining_time // 3600
            remaining_minutes = (estimated_remaining_time % 3600) // 60
            remaining_seconds = estimated_remaining_time % 60

            print(f'Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Remaining Time: {remaining_hours:.0f}h {remaining_minutes:.0f}min {remaining_seconds:.2f}s')

        train_rew += episode_reward
        writer.add_scalar('Reward/train', episode_reward, episode + 1)
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            writer.add_scalar('Loss/train', avg_loss, episode + 1)

        for gb_window in gb_windows:
            if episode + 1 >= gb_start_episodes[gb_window] and episode_reward > best_gb_rews[gb_window]:
                best_gb_rews[gb_window] = episode_reward
                network.save(gb_model_paths[gb_window])
                print(
                    f'New GB{gb_window} model saved from last {gb_window} episodes '
                    f'with training reward: {best_gb_rews[gb_window]}'
                )

        # run validation
        if (episode + 1) % vali_interval == 0:
            print(f'Running validation')
            avg_vali_rew = validate_dqn(env, network, num_validation_episodes)
            print(f'End of validation, Average Validation Reward: {avg_vali_rew}')
            writer.add_scalar('Reward/validation', avg_vali_rew, episode + 1)

            # save model if it is the best so far
            if avg_vali_rew > best_vali_rew:
                best_vali_rew = avg_vali_rew
                network.save(model_path)
                print(f'New best model saved with average validation reward: {best_vali_rew}')


    # compute training time
    end_time = time.time()
    training_time = end_time - start_time
    hours = training_time // 3600
    minutes = (training_time % 3600) // 60
    seconds = training_time % 60

    # print results
    print(f'Training completed, model has been saved with best validation reward: {best_vali_rew}. Training Time: {hours:.0f}h {minutes:.0f}min {seconds:.2f}s')
    writer.add_text('Result/ModelPath', model_path)
    for gb_window in gb_windows:
        writer.add_text(f'Result/GB{gb_window}ModelPath', gb_model_paths[gb_window])
        writer.add_text(f'Result/GB{gb_window}Reward', str(best_gb_rews[gb_window]))
    writer.close()



# PPO validation function
def validate_ppo(env, network, num_validation_episodes):
    vali_rew = 0
    num_episodes = num_validation_episodes

    for episode in range(num_episodes):
        obs = env.reset('validation')
        episode_reward = 0

        for step in range(env.episode_steps):
            act, _, _ = network.select_action(obs, deterministic=False)
            rew, next_obs, done = env.step(act)
            obs = next_obs
            episode_reward += rew

            if done:
                break

        vali_rew += episode_reward
    
    return vali_rew / num_episodes





# PPO training function
def train_ppo(env):

    ##--------Version Information--------##
    network = ppo.PPO_v1(env)
    model_version = 4
    note = "NA"
    ##----------------------------------##

    start_time = time.time()
    num_episodes = network.num_episodes
    output_interval = 20

    # best model tracking
    model_path = os.path.join(model_dir, f'{network.file_name}{model_version}_variant_{env.variant}.pt')
    gb_model_path = os.path.join(model_dir, f'{network.file_name}{model_version}_variant_{env.variant}_GB.pt')
    best_vali_rew = -float('inf')
    best_gb_rew = -float('inf')
    gb_window = max(0, args.gb_window)
    gb_start_episode = max(1, num_episodes - gb_window + 1) if gb_window else num_episodes + 1

    # print info
    print(f'Training {network.network_name} on Variant {env.variant}')

    # validation info
    vali_interval = 100
    num_validation_episodes = 100 # validation episodes in each validation phase

    # file protection
    run_name = f'{network.file_name}{model_version}_variant_{env.variant}'
    log_dir = f'./logs/{run_name}'
    if os.path.exists(model_path) or os.path.exists(gb_model_path) or os.path.exists(log_dir):
        raise FileExistsError(
            f'Model name already exists: {run_name}. '
            f'Please increase model_version before training.'
        )
    
    # tensor-board writer
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Hyperparameters', f'num_episodes: {network.num_episodes}, rollout_steps: {network.rollout_steps}, gamma: {network.gamma}, learning_rate: {network.learning_rate}, clip_epsilon: {network.clip_epsilon}, update_epochs: {network.update_epochs}, entropy_coef: {network.entropy_coef}, value_coef: {network.value_coef}, max_grad_norm: {network.max_grad_norm}', 0)
    writer.add_text('Model_info', f'Model: {run_name}, Variant: {env.variant}', 0)
    writer.add_text('Note', note, 0)
    writer.add_text('GB', f'gb_window: {gb_window}, gb_model_path: {gb_model_path}', 0)

    rollout_obs = []
    rollout_actions = []
    rollout_log_probs = []
    rollout_rewards = []
    rollout_dones = []
    rollout_values = []

    # running
    for episode in range(num_episodes):
        obs = env.reset('training')
        episode_reward = 0

        for step in range(env.episode_steps):
            act, log_prob, value = network.select_action(obs)
            rew, next_obs, done = env.step(act)

            rollout_obs.append(obs)
            rollout_actions.append(act)
            rollout_log_probs.append(log_prob)
            rollout_rewards.append(rew)
            rollout_dones.append(done)
            rollout_values.append(value)

            obs = next_obs
            episode_reward += rew

            if done:
                break

        writer.add_scalar('Reward/train', episode_reward, episode + 1)

        if episode + 1 >= gb_start_episode and episode_reward > best_gb_rew:
            best_gb_rew = episode_reward
            network.save(gb_model_path)
            print(
                f'New GB model saved from last {gb_window} episodes '
                f'with training reward: {best_gb_rew}'
            )

        # update rollout data
        if (episode + 1) % network.rollout_episodes == 0:
            returns, advantages = network.compute_returns(
                rollout_rewards,
                rollout_dones,
                rollout_values
            )

            loss, actor_loss, critic_loss, entropy = network.update(
                rollout_obs,
                rollout_actions,
                rollout_log_probs,
                returns,
                advantages
            )

            writer.add_scalar('Loss/train', loss, episode + 1)
            writer.add_scalar('Loss/actor', actor_loss, episode + 1)
            writer.add_scalar('Loss/critic', critic_loss, episode + 1)
            writer.add_scalar('Entropy/train', entropy, episode + 1)

            rollout_obs = []
            rollout_actions = []
            rollout_log_probs = []
            rollout_rewards = []
            rollout_dones = []
            rollout_values = []

        # print training progress
        if (episode + 1) % output_interval == 0:

            elapsed_time = time.time() - start_time
            episodes_done = episode + 1
            avg_time_per_episode = elapsed_time / episodes_done
            remaining_episodes = num_episodes - episodes_done
            estimated_remaining_time = avg_time_per_episode * remaining_episodes

            remaining_hours = estimated_remaining_time // 3600
            remaining_minutes = (estimated_remaining_time % 3600) // 60
            remaining_seconds = estimated_remaining_time % 60

            print(f'Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Remaining Time: {remaining_hours:.0f}h {remaining_minutes:.0f}min {remaining_seconds:.2f}s')


        # run validation
        if (episode + 1) % vali_interval == 0:
            print(f'Running validation')
            avg_vali_rew = validate_ppo(env, network, num_validation_episodes)
            print(f'End of validation, Average Validation Reward: {avg_vali_rew}')
            writer.add_scalar('Reward/validation', avg_vali_rew, episode + 1)

            # save model if it is the best so far
            if avg_vali_rew > best_vali_rew:
                best_vali_rew = avg_vali_rew
                network.save(model_path)
                print(f'New best model saved with average validation reward: {best_vali_rew}')


    # compute training time
    end_time = time.time()
    training_time = end_time - start_time
    hours = training_time // 3600
    minutes = (training_time % 3600) // 60
    seconds = training_time % 60

    # print results
    print(f'Training completed, model has been saved with best validation reward: {best_vali_rew}. Training Time: {hours:.0f}h {minutes:.0f}min {seconds:.2f}s')
    writer.add_text('Result/ModelPath', model_path)
    writer.add_text('Result/GBModelPath', gb_model_path)
    writer.add_text('Result/GBReward', str(best_gb_rew))
    writer.close()



# TODO: execute training
if __name__ == '__main__':
    train_dqn(env)
