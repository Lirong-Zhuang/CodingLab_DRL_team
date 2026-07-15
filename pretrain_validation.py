# compute average reward per test episode with trained policy
import dqn
import torch
import numpy as np
import pandas as pd
from environment import Environment

def pretrain_validation(env, network, model_path):
    validation_rew = []  # initialize reward tracking
    network.q_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    network.q_network.eval()
    old_epsilon = network.epsilon  # save current epsilon value
    network.epsilon = 0  # set epsilon to 0 for validation

    for i in range(100):  # loop over 100 test episodes
        obs = env.reset('validation')  # get initial obs
        rew_per_eps = 0.

        for j in range(200):  # loop over 200 steps per episode
            act = network.select_action(obs)
            rew, next_obs, _ = env.step(act)  # take one step in the environment
            rew_per_eps += rew  # track rewards
            obs = next_obs  # continue from the new obs
        validation_rew.append(rew_per_eps)

    network.epsilon = old_epsilon
    avg_validation_rew = np.mean(validation_rew)  # compute the average reward per episode
    print(avg_validation_rew)

if __name__ == '__main__':
    variant = 2
    model_path = './models/pretrained_variant_2.pt'
    env = Environment(variant=variant, data_dir='./data')
    network = dqn.DQN_CNN_v2(env)
    pretrain_validation(env, network, model_path)  # test the trained policy