import argparse
from datetime import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.autoencoder.autoencoder_model import build_autoencoder
from environments.rainbow_environment import Environment as RainbowEnvironment


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "autoencoder_models")
DEFAULT_LOG_DIR = os.path.join(SCRIPT_DIR, "autoencoder_logs")


def weighted_reconstruction_loss(reconstructed, target, channel_weights):
    weights = channel_weights.view(1, -1, 1, 1).to(target.device)
    squared_error = F.mse_loss(reconstructed, target, reduction="none")
    return (squared_error * weights).mean()


def read_episode_split(data_dir, variant):
    split_dir = os.path.join(data_dir, f"variant_{variant}")
    train_episodes = pd.read_csv(
        os.path.join(split_dir, "training_episodes.csv")
    ).training_episodes.tolist()
    val_episodes = pd.read_csv(
        os.path.join(split_dir, "validation_episodes.csv")
    ).validation_episodes.tolist()
    test_episodes = pd.read_csv(
        os.path.join(split_dir, "test_episodes.csv")
    ).test_episodes.tolist()
    return train_episodes, val_episodes, test_episodes


def reset_to_episode(env, episode_id):
    env.step_count = 0
    env.agent_loc = (env.vertical_idx_target, env.horizontal_idx_target)
    env.agent_load = 0
    env.item_locs = []
    env.item_times = []
    env.data = pd.read_csv(
        os.path.join(
            env.data_dir,
            f"variant_{env.variant}",
            "episode_data",
            f"episode_{episode_id:03d}.csv",
        ),
        index_col=0,
    )
    return env.get_obs()


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


def collect_episode_observations(env, episode_id, policy, greedy_prob):
    observations = [reset_to_episode(env, episode_id)]

    for _ in range(env.episode_steps):
        action = choose_action(env, policy, greedy_prob)
        _, next_obs, done = env.step(action)
        observations.append(next_obs)

        if done:
            break

    return np.stack(observations).astype(np.float32)


def train_on_observations(model, optimizer, observations, channel_weights, batch_size, device):
    model.train()
    dataset = TensorDataset(torch.from_numpy(observations))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    loss_sum = 0.0
    count = 0
    for (batch,) in loader:
        batch = batch.to(device)
        reconstructed = model(batch)
        loss = weighted_reconstruction_loss(reconstructed, batch, channel_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * batch.size(0)
        count += batch.size(0)

    return loss_sum / count


def evaluate_on_episodes(model, env, episode_ids, args, channel_weights, device):
    model.eval()
    loss_sum = 0.0
    count = 0

    with torch.no_grad():
        for episode_id in episode_ids:
            observations = collect_episode_observations(
                env,
                episode_id,
                args.policy,
                args.greedy_prob,
            )
            dataset = TensorDataset(torch.from_numpy(observations))
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            for (batch,) in loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = weighted_reconstruction_loss(reconstructed, batch, channel_weights)
                loss_sum += loss.item() * batch.size(0)
                count += batch.size(0)

    return loss_sum / count


def format_hyperparameters(args, device, train_episodes, val_episodes, test_episodes):
    values = {
        "env_version": args.env_version,
        "variant": args.variant,
        "model_version": args.model_version,
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "feature_dim": args.feature_dim,
        "encoder_type": args.encoder_type,
        "validation_interval": args.validation_interval,
        "max_train_episodes": args.max_train_episodes,
        "policy": args.policy,
        "greedy_prob": args.greedy_prob,
        "no_shuffle_episodes": args.no_shuffle_episodes,
        "channel_weights": args.channel_weights,
        "seed": args.seed,
        "device": str(device),
        "train_episodes": len(train_episodes),
        "validation_episodes": len(val_episodes),
        "test_episodes": len(test_episodes),
    }
    return "\n".join(f"{key}: {value}" for key, value in values.items())


def format_duration(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:.0f}h {minutes:.0f}min {remaining_seconds:.2f}s"


def train_autoencoder(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"autoencoder_env{args.env_version}_variant{args.variant}"
        f"_v{args.model_version}_{timestamp}"
    )
    tensorboard_log_dir = os.path.join(args.log_dir, run_name)

    train_episodes, val_episodes, test_episodes = read_episode_split(
        args.data_dir,
        args.variant,
    )

    if args.max_train_episodes is not None:
        train_episodes = train_episodes[: args.max_train_episodes]

    print(
        f"Episode split: train={len(train_episodes)} "
        f"validation={len(val_episodes)} test={len(test_episodes)}"
    )

    train_env = RainbowEnvironment(variant=args.variant, data_dir=args.data_dir)
    val_env = RainbowEnvironment(variant=args.variant, data_dir=args.data_dir)
    initial_obs = train_env.reset("training")
    in_channels = initial_obs.shape[0]

    if len(args.channel_weights) != in_channels:
        raise ValueError(
            f"env{args.env_version} observations have {in_channels} channels, "
            f"but --channel_weights has {len(args.channel_weights)} values."
        )

    model = build_autoencoder(
        in_channels=in_channels,
        feature_dim=args.feature_dim,
        encoder_type=args.encoder_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    channel_weights = torch.tensor(args.channel_weights, dtype=torch.float32)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    writer.add_text(
        "Hyperparameters",
        format_hyperparameters(args, device, train_episodes, val_episodes, test_episodes),
        0,
    )

    best_val_loss = float("inf")
    best_state = None
    recent_train_losses = []

    rng = np.random.default_rng(args.seed)
    start_time = time.time()
    total_train_episodes = args.epochs * len(train_episodes)

    for epoch in range(args.epochs):
        if not args.no_shuffle_episodes:
            epoch_train_episodes = rng.permutation(train_episodes).tolist()
        else:
            epoch_train_episodes = list(train_episodes)

        for episode_position, episode_id in enumerate(epoch_train_episodes, start=1):
            observations = collect_episode_observations(
                train_env,
                episode_id,
                args.policy,
                args.greedy_prob,
            )
            train_loss = train_on_observations(
                model,
                optimizer,
                observations,
                channel_weights,
                args.batch_size,
                device,
            )
            recent_train_losses.append(train_loss)
            episodes_seen = (epoch * len(epoch_train_episodes)) + episode_position
            writer.add_scalar("Loss/train", train_loss, episodes_seen)

            should_validate = (
                episode_position % args.validation_interval == 0
                or episode_position == len(epoch_train_episodes)
            )
            if not should_validate:
                continue

            val_loss = evaluate_on_episodes(
                model,
                val_env,
                val_episodes,
                args,
                channel_weights,
                device,
            )
            train_loss_window = float(np.mean(recent_train_losses))
            recent_train_losses = []
            writer.add_scalar("Loss/validation", val_loss, episodes_seen)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "env_version": args.env_version,
                    "in_channels": in_channels,
                    "feature_dim": args.feature_dim,
                    "encoder_type": args.encoder_type,
                    "channel_weights": args.channel_weights,
                    "epoch": epoch + 1,
                    "episodes_seen": (epoch * len(epoch_train_episodes)) + episode_position,
                    "val_loss": val_loss,
                    "train_episode_split": train_episodes,
                    "validation_episode_split": val_episodes,
                    "test_episode_split": test_episodes,
                    "policy": args.policy,
                    "greedy_prob": args.greedy_prob,
                }
            elapsed_time = time.time() - start_time
            avg_time_per_episode = elapsed_time / episodes_seen
            remaining_episodes = total_train_episodes - episodes_seen
            estimated_remaining_time = avg_time_per_episode * remaining_episodes

            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} "
                f"episode {episode_position:03d}/{len(epoch_train_episodes)} "
                f"last_episode={episode_id:03d} "
                f"train_loss={train_loss_window:.6f} "
                f"val_loss={val_loss:.6f} "
                f"Remaining Time: {format_duration(estimated_remaining_time)}"
            )

    training_time = time.time() - start_time
    os.makedirs(args.model_dir, exist_ok=True)
    if args.encoder_type == "cnn":
        file_suffix = f"env{args.env_version}_variant{args.variant}_v{args.model_version}"
    else:
        file_suffix = (
            f"{args.encoder_type}_env{args.env_version}"
            f"_variant{args.variant}_v{args.model_version}"
        )
    autoencoder_path = os.path.join(args.model_dir, f"autoencoder_{file_suffix}.pt")
    encoder_path = os.path.join(args.model_dir, f"encoder_{file_suffix}.pt")

    if not args.overwrite:
        for path in [autoencoder_path, encoder_path]:
            if os.path.exists(path):
                raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    torch.save(best_state, autoencoder_path)
    torch.save(
        {
            "encoder_state_dict": best_state["encoder_state_dict"],
            "encoder_type": best_state["encoder_type"],
            "env_version": best_state["env_version"],
            "in_channels": best_state["in_channels"],
            "feature_dim": best_state["feature_dim"],
            "val_loss": best_state["val_loss"],
        },
        encoder_path,
    )

    writer.add_text("Result/AutoencoderPath", autoencoder_path, 0)
    writer.add_text("Result/EncoderPath", encoder_path, 0)
    writer.add_text("Result/TrainingTime", format_duration(training_time), 0)
    writer.flush()
    writer.close()

    print(f"Saved autoencoder checkpoint to {autoencoder_path}")
    print(f"Saved encoder weights to {encoder_path}")
    print(f"Saved TensorBoard logs to {tensorboard_log_dir}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Training Time: {format_duration(training_time)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_version", type=int, default=11, choices=[11])
    parser.add_argument("--variant", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--model_version", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--feature_channels", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="lenet5",
        choices=["cnn", "lenet5", "alexnet8", "resnet18"],
    )
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--max_train_episodes", type=int, default=None)
    parser.add_argument("--policy", type=str, default="greedy", choices=["random", "greedy", "mixed"])
    parser.add_argument("--greedy_prob", type=float, default=1.0)
    parser.add_argument("--no_shuffle_episodes", action="store_true")
    parser.add_argument(
        "--channel_weights",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.feature_channels is not None:
        args.feature_dim = args.feature_channels
    if args.channel_weights is None:
        args.channel_weights = [4.0, 1.0, 1.0, 4.0, 2.0, 2.0, 2.0]
    return args


if __name__ == "__main__":
    train_autoencoder(parse_args())
