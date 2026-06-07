# compute average reward per test episode with trained policy


from environment import Environment
from environment_v2 import Environment_v2
from environment_v3 import Environment_v3
from environment_v4 import Environment_v4
from environment_v5 import Environment_v5
from environment_v6 import Environment_v6
from environment_v7 import Environment_v7
from environment_v8 import Environment_v8
import torch
import dqn
import rainbow_dqn


def test_policy(env):
    test_rew = 0.  # initialize reward tracking

    # network
    network = rainbow_dqn.DQN_v8(env)
    network.q_network.load_state_dict(torch.load('./models/DQN_v8.5.4_variant_2.pt', map_location=network.device))
    network.q_network.eval()
    network.epsilon = 0

    for i in range(100):  # loop over 100 test episodes
        obs = env.reset('testing')  # get initial obs

        for j in range(200):  # loop over 200 steps per episode
            act = network.select_action(obs)  # TODO: get action for the obs from your trained policy
            rew, next_obs, _ = env.step(act)  # take one step in the environment
            test_rew += rew  # track rewards
            obs = next_obs  # continue from the new obs

    avg_test_rew = test_rew / 100  # compute the average reward per episode

    print(avg_test_rew)  # print the result


if __name__ == '__main__':

    data_dir = './data' # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
    variant = 2  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
    env = Environment_v5(variant=variant, data_dir=data_dir)  # initialize the environment

    test_policy(env)  # test the trained policy
