import time
from torch.utils.tensorboard import SummaryWriter
import dqn
# TODO: parse arguments
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--variant', type=int, default=0, choices=[0, 1, 2])
parser.add_argument('--seed', type=int, default=2026) # seed can be any other number
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


# initialize environment
from environment import Environment

data_dir = args.data_dir  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = args.variant  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
env = Environment(variant, data_dir)
model_dir = './models'


# validation function
def validate(env, network, num_validation_episodes):
    old_epsilon = network.epsilon  # save current epsilon value
    network.epsilon = 0  # set epsilon to 0 for validation

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

    avg_reward = vali_rew / num_episodes
    return avg_reward

# train function
def train(env):

    ##--------Version Information--------##
    network = dqn.DQN_v4(env)
    model_version = 3
    # note = "Architecture: 54 64 64 5"
    note = "N/A"
    ##----------------------------------##

    start_time = time.time()
    train_rew = 0  # initialize reward tracking
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
            avg_vali_rew = validate(env, network, num_validation_episodes)
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
    train(env)


