# compute average reward per test episode with trained policy


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
import csv
import os
import torch
import dqn
import rainbow_dqn


MODEL_NAME = 'DQN_v11.9.6_variant_0'
MODEL_PATH = f'./models/{MODEL_NAME}.pt'
VARIANT = 0
DATA_DIR = './data'
NUM_TEST_EPISODES = 100
RESULT_DIR = './test_result'
RESULT_FILE_NAME = 'Test_Results_DQN_v11.9.6_variant_0.csv'


def test_policy(env):
    test_rew = 0.  # initialize reward tracking
    episode_results = []

    # network
    network = rainbow_dqn.DQN_v11(env)
    network.q_network.load_state_dict(torch.load(MODEL_PATH, map_location=network.device))
    network.q_network.eval()
    network.epsilon = 0

    for i in range(NUM_TEST_EPISODES):  # loop over 100 test episodes
        obs = env.reset('testing')  # get initial obs
        episode_rew = 0.

        for j in range(env.episode_steps):  # loop over all steps per episode
            act = network.select_action(obs)  # TODO: get action for the obs from your trained policy
            rew, next_obs, done = env.step(act)  # take one step in the environment
            test_rew += rew  # track rewards
            episode_rew += rew
            obs = next_obs  # continue from the new obs

            if done:
                break

        episode_results.append(episode_rew)

    avg_test_rew = test_rew / NUM_TEST_EPISODES  # compute the average reward per episode
    result_path = save_test_results(env, episode_results, avg_test_rew)

    print(f'Average Test Reward of Model {MODEL_NAME}: {avg_test_rew}')  # print the result
    print(f'Test results saved to: {result_path}')


def save_test_results(env, episode_results, avg_test_rew):
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_path = os.path.join(RESULT_DIR, RESULT_FILE_NAME)

    with open(result_path, 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        for episode_idx, episode_rew in enumerate(episode_results, start=1):
            writer.writerow([f'Episode {episode_idx} : {episode_rew:.2f}'])
            
        writer.writerow([f'Max Reward : {max(episode_results):.2f}'])
        writer.writerow([f'Average Reward : {avg_test_rew:.2f}'])
        
    return result_path


if __name__ == '__main__':

    env = Environment_v9(variant=VARIANT, data_dir=DATA_DIR)  # initialize the environment

    test_policy(env)  # test the trained policy
