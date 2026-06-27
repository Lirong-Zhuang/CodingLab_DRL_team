import argparse
import os
import random

import numpy as np
import torch

from environment import Environment
from ppo import PPO_v1


def train_ppo(env, agent, model_path, skip_bc=False):
    if not skip_bc and agent.bc_episodes > 0:
        print("Behavioural cloning warm-start...")
        agent.run_bc_phase()
        agent.save(model_path)

    best_reward = -float("inf")

    print("Behavioural cloning done. Starting PPO...\n")

    for episode in range(agent.num_episodes):
        agent.decay_entropy(episode)

        obs = env.reset("training")

        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        episode_reward = 0.0

        for _ in range(env.episode_steps):
            action, log_prob, value = agent.select_action(
                obs,
                deterministic=False
            )

            reward, next_obs, done = env.step(action)

            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            episode_reward += reward
            obs = next_obs

            if done:
                break

        returns, advantages = agent.compute_returns(
            rewards=rewards,
            dones=dones,
            values=values
        )

        loss, actor_loss, critic_loss, entropy = agent.update(
            obs=observations,
            actions=actions,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages
        )

        if (episode + 1) % 100 == 0:
            validation_reward = evaluate_agent(env, agent, num_episodes=100, mode="validation")
            if validation_reward > best_reward:
                best_reward = validation_reward
                agent.save(model_path)

            print(
                f"Episode {episode + 1}/{agent.num_episodes}  "
                f"Reward: {episode_reward:.2f}  "
                f"Validation: {validation_reward:.2f}  "
                f"Best validation: {best_reward:.2f}  "
                f"Entropy: {agent.entropy_coef:.4f}  "
                f"Loss: {loss:.4f}"
            )

    print("\nTraining finished.")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Model saved to: {model_path}")


def evaluate_agent(env, agent, num_episodes=100, mode="validation"):
    rewards = []
    was_training = agent.network.training
    agent.network.eval()

    for _ in range(num_episodes):
        obs = env.reset(mode)
        episode_reward = 0.0
        done = 0
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            reward, obs, done = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)

    if was_training:
        agent.network.train()
    return float(np.mean(rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--variant", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--feature_mode", type=str, default="all")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--bc_episodes", type=int, default=None)
    parser.add_argument("--bc_epochs", type=int, default=None)
    parser.add_argument("--skip_bc", action="store_true")

    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = Environment(
        variant=args.variant,
        data_dir=args.data_dir
    )

    env.feature_mode = args.feature_mode

    # Reward shaping during training
    env.reward_shaping = True

    os.makedirs("checkpoints", exist_ok=True)

    if args.model_path is None:
        model_path = f"checkpoints/PPO_v1_variant_{args.variant}.pt"
    else:
        model_path = args.model_path

    agent = PPO_v1(env)
    if args.episodes is not None:
        agent.num_episodes = args.episodes
    if args.bc_episodes is not None:
        agent.bc_episodes = args.bc_episodes
    if args.bc_epochs is not None:
        agent.bc_epochs = args.bc_epochs

    train_ppo(
        env=env,
        agent=agent,
        model_path=model_path,
        skip_bc=args.skip_bc
    )
