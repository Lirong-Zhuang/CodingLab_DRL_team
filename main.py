import time
from torch.utils.tensorboard import SummaryWriter
import dqn
import ppo
# TODO: parse arguments
import argparse
import rainbow_dqn

parser = argparse.ArgumentParser()

parser.add_argument('--variant', type=int, default=0, choices=[0, 1, 2])
parser.add_argument('--seed', type=int, default=777) # seed can be any other number
parser.add_argument('--data_dir', type=str, default='./data')

args = parser.parse_args()


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

data_dir = args.data_dir  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = args.variant  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
# env = Environment(variant, data_dir)
# env = Environment_v2(variant, data_dir)
env = Environment_v4(variant, data_dir)
model_dir = './models'


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
    network = rainbow_dqn.DQN_v8(env)
    model_version = 2
    # note = "Architecture: 54 64 64 5"
    note = "NA"
    ##----------------------------------##

    start_time = time.time()
    train_rew = 0  # initialize reward tracking
    num_episodes = network.num_episodes
    output_interval = 20

    # best model tracking
    model_path = os.path.join(model_dir, f'{network.file_name}{env.env_name}{model_version}_variant_{env.variant}.pt')
    best_vali_rew = -float('inf')

    # print info
    print(f'Training {network.network_name} on Variant {env.variant}')

    # validation info
    vali_interval = 100
    num_validation_episodes = 100 # validation episodes in each validation phase

    # file protection
    run_name = f'{network.file_name}{env.env_name}{model_version}_variant_{env.variant}'
    log_dir = f'./logs/{run_name}'
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
    network = ppo.PPO_v1(env)
    model_version = 3
    note = "Test to avoid stop greedy action"
    ##----------------------------------##

    start_time = time.time()
    num_episodes = network.num_episodes
    output_interval = 20

    # best model tracking
    model_path = os.path.join(model_dir, f'{network.file_name}{model_version}_variant_{env.variant}.pt')
    best_vali_rew = -float('inf')

    # print info
    print(f'Training {network.network_name} on Variant {env.variant}')

    # validation info
    vali_interval = 100
    num_validation_episodes = 100 # validation episodes in each validation phase

    # file protection
    run_name = f'{network.file_name}{model_version}_variant_{env.variant}'
    log_dir = f'./logs/{run_name}'
    if os.path.exists(model_path) or os.path.exists(log_dir):
        raise FileExistsError(
            f'Model name already exists: {run_name}. '
            f'Please increase model_version before training.'
        )
    
    # tensor-board writer
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Hyperparameters', f'num_episodes: {network.num_episodes}, rollout_steps: {network.rollout_steps}, gamma: {network.gamma}, learning_rate: {network.learning_rate}, clip_epsilon: {network.clip_epsilon}, update_epochs: {network.update_epochs}, entropy_coef: {network.entropy_coef}, value_coef: {network.value_coef}, max_grad_norm: {network.max_grad_norm}', 0)
    writer.add_text('Model_info', f'Model: {run_name}, Variant: {env.variant}', 0)
    writer.add_text('Note', note, 0)

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
    writer.close()



# TODO: execute training
if __name__ == '__main__':
    train_dqn(env)


