import argparse
import os
import sys

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment_v5 import Environment_v5


def choose_greedy_action(env):
    if env.agent_load > 0:
        target = env.target_loc
    elif env.item_locs:
        distances = [
            abs(env.agent_loc[0] - item_loc[0]) + abs(env.agent_loc[1] - item_loc[1])
            for item_loc in env.item_locs
        ]
        target = env.item_locs[int(np.argmin(distances))]
    else:
        return 0

    row, col = env.agent_loc
    target_row, target_col = target
    candidates = []

    if target_row < row:
        candidates.append(1)
    if target_col > col:
        candidates.append(2)
    if target_row > row:
        candidates.append(3)
    if target_col < col:
        candidates.append(4)

    for action in candidates:
        if action == 1:
            new_loc = (row - 1, col)
        elif action == 2:
            new_loc = (row, col + 1)
        elif action == 3:
            new_loc = (row + 1, col)
        else:
            new_loc = (row, col - 1)

        if new_loc in env.eligible_cells:
            return action

    return np.random.randint(5)


def choose_action(env, policy, greedy_prob):
    if policy == "random":
        return np.random.randint(5)
    if policy == "greedy":
        return choose_greedy_action(env)
    if np.random.rand() < greedy_prob:
        return choose_greedy_action(env)
    return np.random.randint(5)


def collect_observations(args):
    np.random.seed(args.seed)
    env = Environment_v5(variant=args.variant, data_dir=args.data_dir)
    observations = []

    for episode in range(args.num_episodes):
        obs = env.reset("training")
        observations.append(obs)

        for _ in range(env.episode_steps):
            action = choose_action(env, args.policy, args.greedy_prob)
            _, next_obs, done = env.step(action)
            observations.append(next_obs)

            if done:
                break

        if args.log_interval and (episode + 1) % args.log_interval == 0:
            print(f"Collected {episode + 1}/{args.num_episodes} episodes")

    obs_array = np.stack(observations).astype(np.float32)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez_compressed(
        args.output_path,
        obs=obs_array,
        variant=np.array(args.variant, dtype=np.int64),
        env_version=np.array(5, dtype=np.int64),
        policy=np.array(args.policy),
    )

    print(f"Saved observations to {args.output_path}")
    print(f"Observation array shape: {obs_array.shape}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_episodes", type=int, default=800)
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/autoencoder_env5_variant0_obs.npz",
    )
    parser.add_argument("--policy", type=str, default="mixed", choices=["random", "greedy", "mixed"])
    parser.add_argument("--greedy_prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    collect_observations(parse_args())
