# compute average reward per test episode with trained policy


from environments.environment import Environment


def test_policy(env):
    test_rew = 0.  # initialize reward tracking
    episode_results = []

    for i in range(100):  # loop over 100 test episodes
        obs = env.reset('testing')  # get initial obs

        for j in range(200):  # loop over 200 steps per episode
            act = network.select_action(obs)  # TODO: get action for the obs from your trained policy
            rew, next_obs, done = env.step(act)  # take one step in the environment
            test_rew += rew  # track rewards
            episode_rew += rew
            obs = next_obs  # continue from the new obs

            if done:
                break

        episode_results.append(episode_rew)       

    avg_test_rew = test_rew / 100  # compute the average reward per episode
    result_path = save_test_results(env, episode_results, avg_test_rew)

    print(f'Average Test Reward of Model {MODEL_NAME}: {avg_test_rew}')  # print the result
    print(f'Test results saved to: {result_path}')



if __name__ == '__main__':

    data_dir = ...  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
    variant = ...  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
    env = Environment(variant=variant, data_dir=data_dir)  # initialize the environment

    test_policy(env)  # test the trained policy
