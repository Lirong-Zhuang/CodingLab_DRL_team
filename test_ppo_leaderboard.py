import argparse
import os
import shutil
import numpy as np
import pandas as pd
import torch

from environment import Environment
from ppo import PPO_v1


def prepare_leaderboard_data(base_data_dir, leaderboard_dir, output_dir, variant):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    shutil.copytree(base_data_dir, output_dir)

    variant_dir = os.path.join(output_dir, f"variant_{variant}")
    episode_data_dir = os.path.join(variant_dir, "episode_data")

    os.makedirs(episode_data_dir, exist_ok=True)

    source_dir = os.path.join(leaderboard_dir, f"variant_{variant}")

    episode_ids = []

    for i in range(100):
        episode_id = i
        episode_ids.append(episode_id)

        src = os.path.join(source_dir, f"episode_{i:03d}.csv")
        dst = os.path.join(episode_data_dir, f"episode_{i:03d}.csv")

        shutil.copy(src, dst)

    test_csv = pd.DataFrame({"test_episodes": episode_ids})
    test_csv.to_csv(
        os.path.join(variant_dir, "test_episodes.csv"),
        index=False
    )

    return output_dir


def load_ppo_checkpoint(agent, model_path):
    checkpoint = torch.load(model_path, map_location=agent.device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        agent.network.load_state_dict(checkpoint["model_state_dict"])
    else:
        agent.network.load_state_dict(checkpoint)

    agent.network.eval()


def evaluate_ppo(env, agent, num_episodes=100):
    rewards = []

    for ep in range(num_episodes):
        obs = env.reset("testing")
        episode_reward = 0.0

        for step in range(env.episode_steps):
            action, _, _ = agent.select_action(obs, deterministic=True)
            reward, next_obs, done = env.step(action)

            episode_reward += reward
            obs = next_obs

            if done:
                break

        rewards.append(episode_reward)
        print(f"Episode {ep + 1:03d}: reward = {episode_reward:.2f}")

    rewards = np.array(rewards)

    print("\n==============================")
    print("PPO Leaderboard Test Results")
    print("==============================")
    print(f"Average reward : {rewards.mean():.2f}")
    print(f"Std reward     : {rewards.std():.2f}")
    print(f"Min reward     : {rewards.min():.2f}")
    print(f"Max reward     : {rewards.max():.2f}")

    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--variant", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--base_data_dir", type=str, default="./data")
    parser.add_argument("--leaderboard_dir", type=str, default="./test_episodes")
    parser.add_argument("--output_data_dir", type=str, default="./data_leaderboard")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="ppo_leaderboard_results.csv")

    args = parser.parse_args()

    data_dir = prepare_leaderboard_data(
        base_data_dir=args.base_data_dir,
        leaderboard_dir=args.leaderboard_dir,
        output_dir=args.output_data_dir,
        variant=args.variant
    )

    env = Environment(
        variant=args.variant,
        data_dir=data_dir
    )

    agent = PPO_v1(env)

    load_ppo_checkpoint(
        agent=agent,
        model_path=args.model_path
    )

    rewards = evaluate_ppo(
        env=env,
        agent=agent,
        num_episodes=100
    )

    results = pd.DataFrame({
        "episode": range(1, len(rewards) + 1),
        "reward": rewards
    })

    results.to_csv(args.output_csv, index=False)

    print(f"\nSaved results to: {args.output_csv}")