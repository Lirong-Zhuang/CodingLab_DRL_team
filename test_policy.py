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
from environment_v91 import Environment_v91
from environment_v10 import Environment_v10
from environment_v11 import Environment_v11
from environment_v12 import Environment_v12
from environment_v13 import Environment_v13
from environment_v16 import Environment_v16
from environment_v17 import Environment_v17
from environment_vaschley import Environment_vaschley
import csv
import os
import torch
import dqn
import rainbow_dqn
from aschley_dqn import AschleyCNNv2


# MODEL_NAME = 'DQN_v8.13.2_variant_2'
MODEL_NAME = 'DQN_v8.5.30_variant_2'
MODEL_PATH = f'./models2/{MODEL_NAME}.pt'
VARIANT = 2
DATA_DIR = './data'
FINAL_TEST_DATA_DIR = './data/final_test_episodes'
NUM_TEST_EPISODES = 100
RESULT_DIR = './final_test_result'
RESULT_FILE_NAME = f'Test_Results_{MODEL_NAME}.csv'


def load_network(env):
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    if 'q_head.value_stream.0.weight' in state_dict:
        if 'encoder.scene_encoder.conv1.weight' in state_dict:
            encoder_type = 'resnet18'
        elif 'encoder.cnn.0.weight' in state_dict:
            first_encoder_channels = state_dict['encoder.cnn.0.weight'].shape[0]
            encoder_types = {6: 'lenet5', 16: 'cnn', 32: 'alexnet8'}
            if first_encoder_channels not in encoder_types:
                raise ValueError(
                    f'Unsupported DQN_v11 encoder width {first_encoder_channels} '
                    f'in {MODEL_PATH}'
                )
            encoder_type = encoder_types[first_encoder_channels]
        else:
            raise ValueError(f'Cannot identify DQN_v11 encoder in {MODEL_PATH}')

        network = rainbow_dqn.DQN_v11(env, encoder_type=encoder_type)
        architecture = f'DQN_v11/{encoder_type}'
    elif 'feature.0.weight' in state_dict:
        feature_width = state_dict['feature.0.weight'].shape[0]
        if feature_width == 32:
            network = rainbow_dqn.DQN_v8(env)
            architecture = 'DQN_v8'
        elif feature_width == 64:
            network = rainbow_dqn.DQN_v8_old(env)
            architecture = 'DQN_v8_old'
        else:
            raise ValueError(
                f'Unsupported DQN_v8 feature width {feature_width} in {MODEL_PATH}'
            )
    else:
        sample_keys = ', '.join(list(state_dict)[:5])
        raise ValueError(
            f'Unsupported checkpoint architecture in {MODEL_PATH}. Keys: {sample_keys}'
        )

    network.q_network.load_state_dict(state_dict)
    network.q_network.eval()
    network.epsilon = 0
    print(f'Loaded {MODEL_NAME} as {architecture}')
    return network


def test_policy(env):
    test_rew = 0.  # initialize reward tracking
    episode_results = []

    # network
    network = load_network(env)

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

    env = Environment_v5(
        variant=VARIANT,
        data_dir=DATA_DIR,
        test_data_dir=FINAL_TEST_DATA_DIR,
    )

    test_policy(env)  # test the trained policy
