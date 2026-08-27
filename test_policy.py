# compute average reward per test episode with trained policy
import dqn
import torch
import numpy as np
import pandas as pd
from environment import Environment



def test_policy(env, network, model_path):
    test_rew = []  # initialize reward tracking
    network.q_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    network.q_network.eval()
    old_epsilon = network.epsilon  # save current epsilon value
    network.epsilon = 0  # set epsilon to 0 for validation

    for i in range(100):  # loop over 100 test episodes
        obs = env.reset('testing')  # get initial obs
        rew_per_eps = 0.

        for j in range(200):  # loop over 200 steps per episode
            act = network.select_action(obs)
            rew, next_obs, _ = env.step(act)  # take one step in the environment
            rew_per_eps += rew  # track rewards
            obs = next_obs  # continue from the new obs
        test_rew.append(rew_per_eps)

    network.epsilon = old_epsilon
    avg_test_rew = np.mean(test_rew)  # compute the average reward per episode
    df = pd.DataFrame({
        'episode': range(1, 101),
        'reward': test_rew
    })
    df.loc[len(df)] = ['mean', avg_test_rew]

    csv_path = f'./test_results/test_results_variant_{env.variant}.csv'
    df.to_csv(csv_path, index=False)
    print(f'Results saved to {csv_path}')
    print(avg_test_rew)  # print the result


if __name__ == '__main__':
    variant = 1
    model_path = './models/DQN_CNN_v2.2_variant_1.pt'
    env = Environment(variant=variant, data_dir='./data')
    network = dqn.DQN_CNN_v2(env)
    #test_env = Environment(variant=variant, data_dir='./data')  # initialize the environment
    test_policy(env, network, model_path)  # test the trained policy







