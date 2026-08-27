import argparse
import time
from torch.utils.tensorboard import SummaryWriter

from algorithms import dqn, ppo, rainbow
from environments.dqn_environment import Environment as DQNEnvironment
from environments.ppo_environment import Environment as PPOEnvironment
from environments.rainbow_environment import Environment as RainbowEnvironment


# Edit these values before running this file directly.
VARIANT = 1
SEED = 777
DATA_DIR = './data'
ALGORITHM = "rainbow"
MODEL_VERSION = 1
NUM_EPISODES = 10000
PPO_ARCHITECTURE = "hybrid"
BC_EPISODES = 2000
BC_EPOCHS = 20


class Config:
    variant = VARIANT
    seed = SEED
    data_dir = DATA_DIR
    algorithm = ALGORITHM
    model_version = MODEL_VERSION
    num_episodes = NUM_EPISODES
    ppo_architecture = PPO_ARCHITECTURE
    bc_episodes = BC_EPISODES
    bc_epochs = BC_EPOCHS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, default=VARIANT, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument(
        "--algorithm",
        type=str,
        default=ALGORITHM,
        choices=["dqn", "rainbow", "ppo"],
    )
    parser.add_argument("--model_version", type=int, default=MODEL_VERSION)
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    parser.add_argument(
        "--ppo_architecture",
        type=str,
        default=PPO_ARCHITECTURE,
        choices=["hybrid", "cnn", "mlp"],
    )
    parser.add_argument("--bc_episodes", type=int, default=BC_EPISODES)
    parser.add_argument("--bc_epochs", type=int, default=BC_EPOCHS)
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

# TensorFlow is not used by the PyTorch training pipeline. Importing it can
# conflict with the CUDA libraries loaded by PyTorch on the cluster.
# import tensorflow as tf
# tf.random.set_seed(seed)

import torch
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

data_dir = args.data_dir  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = args.variant  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
model_dir = './outputs/checkpoints'
log_root_dir = './outputs/tensorboard'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_root_dir, exist_ok=True)


def build_env(algorithm, variant, data_dir):
    if algorithm == "ppo":
        return PPOEnvironment(variant, data_dir)
    if algorithm == "dqn":
        return DQNEnvironment(variant, data_dir)
    return RainbowEnvironment(variant, data_dir)


def build_dqn_network(algorithm, env):
    if algorithm == "dqn":
        if env.variant == 2:
            return dqn.DQN_CNN_v2_variant_2_action_masking(env)
        return dqn.DQN_CNN_v2(env)
    return rainbow.rainbow_dqn(env)


def build_run_name(algorithm, model_version, variant):
    prefix = 'rainbow_dqn' if algorithm == 'rainbow' else algorithm
    return f'{prefix}_v{model_version}_variant_{variant}'


env = build_env(args.algorithm, variant, data_dir)


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
    network = build_dqn_network(args.algorithm, env)
    if args.num_episodes is not None:
        network.num_episodes = args.num_episodes
        if hasattr(network, 'epsilon_decay_steps'):
            network.epsilon_decay_steps = network.num_episodes * 0.8
    model_version = args.model_version
    note = f'algorithm: {args.algorithm}, environment: {env.__class__.__name__}'
    ##----------------------------------##

    start_time = time.time()
    train_rew = 0  # initialize reward tracking
    num_episodes = network.num_episodes
    output_interval = 20

    # best model tracking
    run_name = build_run_name(args.algorithm, model_version, env.variant)
    model_path = os.path.join(model_dir, f'{run_name}.pt')
    best_vali_rew = -float('inf')
    # print info
    print(f'Training {network.network_name} on Variant {env.variant}')

    # validation info
    vali_interval = 100
    num_validation_episodes = 100 # validation episodes in each validation phase

    # file protection
    log_dir = os.path.join(log_root_dir, run_name)
    if os.path.exists(model_path) or os.path.exists(log_dir):
        raise FileExistsError(
            f'Model name already exists: {run_name}. '
            f'Please increase model_version before training.'
        )

    # tensor-board writer
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Hyperparameters', f'num_episodes: {network.num_episodes}, batch_size: {network.batch_size}, gamma: {network.gamma}, learning_rate: {network.learning_rate}, epsilon_start: {network.epsilon_start}, epsilon_end: {network.epsilon_end}, epsilon_decay_steps: {network.epsilon_decay_steps}, target_update_freq: {network.target_update_freq}', 0)
    writer.add_text('Model_info', f'Model: {run_name}, Variant: {env.variant}', 0)
    writer.add_text('Note', note, 0)

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
    network = ppo.PPO_v1(env, architecture=args.ppo_architecture)
    network.num_episodes = args.num_episodes
    network.bc_episodes = args.bc_episodes
    network.bc_epochs = args.bc_epochs
    rollout_episodes = 10
    model_version = args.model_version
    note = (
        "Lejla PPO with behavioural-cloning warm start; "
        f"architecture: {args.ppo_architecture}"
    )
    ##----------------------------------##

    start_time = time.time()
    num_episodes = network.num_episodes
    output_interval = 20

    # best model tracking
    run_name = build_run_name(args.algorithm, model_version, env.variant)
    model_path = os.path.join(model_dir, f'{run_name}.pt')
    best_vali_rew = -float('inf')

    # print info
    print(f'Training {network.network_name} on Variant {env.variant}')

    # validation info
    vali_interval = 100
    num_validation_episodes = 100 # validation episodes in each validation phase

    # file protection
    log_dir = os.path.join(log_root_dir, run_name)
    if os.path.exists(model_path) or os.path.exists(log_dir):
        raise FileExistsError(
            f'Model name already exists: {run_name}. '
            f'Please increase model_version before training.'
        )
    
    # tensor-board writer
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Hyperparameters', f'num_episodes: {network.num_episodes}, rollout_episodes: {rollout_episodes}, gamma: {network.gamma}, learning_rate: {network.learning_rate}, clip_epsilon: {network.clip_epsilon}, update_epochs: {network.update_epochs}, entropy_coef: {network.entropy_coef}, value_coef: {network.value_coef}, max_grad_norm: {network.max_grad_norm}', 0)
    writer.add_text('Model_info', f'Model: {run_name}, Variant: {env.variant}', 0)
    writer.add_text('Note', note, 0)

    rollout_obs = []
    rollout_actions = []
    rollout_log_probs = []
    rollout_rewards = []
    rollout_dones = []
    rollout_values = []

    if network.bc_episodes > 0:
        network.run_bc_phase()

    # running
    for episode in range(num_episodes):
        network.decay_entropy(episode)
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

        # update rollout data
        if (episode + 1) % rollout_episodes == 0:
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
    writer.close()



# TODO: execute training
if __name__ == '__main__':
    if args.algorithm == "ppo":
        train_ppo(env)
    else:
        train_dqn(env)
