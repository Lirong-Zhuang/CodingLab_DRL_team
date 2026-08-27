# compute average reward per test episode with trained policy


import argparse
import torch

from algorithms import dqn, ppo, rainbow
from environments.dqn_environment import Environment as DQNEnvironment
from environments.ppo_environment import Environment as PPOEnvironment
from environments.rainbow_environment import Environment as RainbowEnvironment


def load_state_dict(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict'], checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict'], checkpoint
    return checkpoint, checkpoint


def load_policy(algorithm, env, model_path, ppo_architecture=None):
    state_dict, checkpoint = load_state_dict(model_path)

    if algorithm == 'ppo':
        architecture = ppo_architecture
        if architecture is None and isinstance(checkpoint, dict):
            architecture = checkpoint.get('architecture')
        network = ppo.PPO_v1(env, architecture=architecture or 'hybrid')
        network.network.load_state_dict(state_dict)
        network.network.eval()
        return network

    if algorithm == 'dqn':
        if env.variant == 2:
            network = dqn.DQN_CNN_v2_variant_2_action_masking(env)
        else:
            network = dqn.DQN_CNN_v2(env)
    else:
        network = rainbow.rainbow_dqn(env)

    network.q_network.load_state_dict(state_dict)
    network.q_network.eval()
    network.epsilon = 0
    return network


def build_environment(algorithm, variant, data_dir):
    if algorithm == 'ppo':
        return PPOEnvironment(variant, data_dir)
    if algorithm == 'dqn':
        return DQNEnvironment(variant, data_dir)
    return RainbowEnvironment(variant, data_dir)


def test_policy(env, network, algorithm, num_test_episodes=100):
    test_rew = 0.  # initialize reward tracking

    for i in range(num_test_episodes):  # loop over test episodes
        obs = env.reset('testing')  # get initial obs

        for j in range(env.episode_steps):  # loop over all steps per episode
            if algorithm == 'ppo':
                act, _, _ = network.select_action(obs, deterministic=True)
            else:
                act = network.select_action(obs)
            rew, next_obs, done = env.step(act)  # take one step in the environment
            test_rew += rew  # track rewards
            obs = next_obs  # continue from the new obs

            if done:
                break

    avg_test_rew = test_rew / num_test_episodes  # compute the average reward per episode

    print(avg_test_rew)  # print the result
    return avg_test_rew


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=['dqn', 'rainbow', 'ppo'], required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--variant', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--num_test_episodes', type=int, default=100)
    parser.add_argument('--ppo_architecture', choices=['hybrid', 'cnn', 'mlp'])
    args = parser.parse_args()

    env = build_environment(args.algorithm, args.variant, args.data_dir)  # initialize the environment
    network = load_policy(args.algorithm, env, args.model_path, args.ppo_architecture)

    test_policy(env, network, args.algorithm, args.num_test_episodes)  # test the trained policy
