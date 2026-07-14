import argparse
from copy import deepcopy
import heapq
import os

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import torch
import torch.nn as nn

import dqn
import rainbow_dqn
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
from environment_v13 import Environment_v13
from environment_v14 import Environment_v14
from environment_v15 import Environment_v15


ACTION_NAMES = {
    0: "stay",
    1: "up",
    2: "right",
    3: "down",
    4: "left",
}


# Edit these values, then run this file directly.
POLICY = "model"  # "greedy", "greedy_astar", or "model"
MODEL_PATH = "./models2/DQN_v8.5.31_variant_2.pt"  # e.g. "./models/DQN_v8.5.4_variant_2.pt"; only needed for POLICY = "model"
VARIANT = 2
ENV_VERSION = 5
NETWORK_VERSION = 8
DATA_DIR = "./data"
EPISODE_ID = "096"
MAX_STEPS = None
INTERVAL = 350
SAVE_PATH = "./videos/8.5.31_test_096.mp4"
VIDEO_FPS = 4
VIDEO_DPI = 120
FFMPEG_PATH = None  # e.g. "/opt/homebrew/bin/ffmpeg"; leave None to use system PATH
CPU = False
SHOW = True
REPEAT = True
Q_LOG_STEPS = ""  # e.g. "169-174" or "169,170,171"; empty string disables printing
ROBOT_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "assets", "robot_agent_256.png")


ENV_CLASSES = {
    1: Environment,
    2: Environment_v2,
    3: Environment_v3,
    4: Environment_v4,
    5: Environment_v5,
    6: Environment_v6,
    7: Environment_v7,
    8: Environment_v8,
    9: Environment_v9,
    10: Environment_v10,
    11: Environment_v11,
    12: Environment_v12,
    13: Environment_v13,
    14: Environment_v14,
    15: Environment_v15,
}


NETWORK_CLASSES = {
    5: dqn.DQN_v5,
    6: rainbow_dqn.DQN_v6,
    7: rainbow_dqn.DQN_v7,
    8: rainbow_dqn.DQN_v8,
    9: rainbow_dqn.DQN_v9,
    10: rainbow_dqn.DQN_v10,
    11: rainbow_dqn.DQN_v11,
}


_ROBOT_IMAGE = None


def robot_image():
    global _ROBOT_IMAGE
    if _ROBOT_IMAGE is None:
        _ROBOT_IMAGE = plt.imread(ROBOT_IMAGE_PATH)
    return _ROBOT_IMAGE


class LegacyRainbowCNNQNetwork64(nn.Module):
    def __init__(self, in_channels, act_dim, num_atoms):
        super().__init__()
        self.act_dim = act_dim
        self.num_atoms = num_atoms

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.feature = nn.Sequential(
            nn.Linear(32 * 3 * 3, 64),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_atoms),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim * num_atoms),
        )

    def forward(self, x):
        features = self.cnn(x)
        features = self.feature(features)

        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.act_dim, self.num_atoms)

        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(logits, dim=-1)

    def reset_noise(self):
        pass


def adapt_legacy_network_if_needed(network, network_version, state_dict, device):
    if network_version != 8:
        return
    feature_weight = state_dict.get("feature.0.weight")
    current_feature_weight = network.q_network.state_dict().get("feature.0.weight")
    if feature_weight is None or current_feature_weight is None:
        return
    if feature_weight.shape == current_feature_weight.shape:
        return
    if feature_weight.shape == torch.Size([64, 288]):
        network.q_network = LegacyRainbowCNNQNetwork64(
            network.in_channels,
            network.act_dim,
            network.num_atoms,
        ).to(device)
        network.target_network = LegacyRainbowCNNQNetwork64(
            network.in_channels,
            network.act_dim,
            network.num_atoms,
        ).to(device)
        return
    raise RuntimeError(
        "Unsupported DQN_v8 checkpoint shape: "
        f"feature.0.weight is {tuple(feature_weight.shape)}, "
        f"current model expects {tuple(current_feature_weight.shape)}"
    )


class GreedyPolicy:
    def __init__(self, env):
        self.env = env

    def select_action(self, obs):
        if self.env.agent_load == self.env.agent_capacity:
            return self.action_towards(self.env.target_loc)

        best_item = None
        best_profit = 0
        for item_loc, item_time in zip(self.env.item_locs, self.env.item_times):
            distance_to_item = self.shortest_distance(self.env.agent_loc, item_loc)
            if distance_to_item is None:
                continue

            time_left = self.env.max_response_time - item_time
            if distance_to_item > time_left:
                continue

            distance_to_target = self.shortest_distance(item_loc, self.env.target_loc)
            if distance_to_target is None:
                continue

            profit = self.env.reward - distance_to_item - distance_to_target
            if profit > best_profit:
                best_profit = profit
                best_item = item_loc

        if best_item is None:
            return 0
        return self.action_towards(best_item)

    def action_towards(self, target_loc):
        if self.env.agent_loc == target_loc:
            return 0

        next_loc = self.next_cell_on_shortest_path(self.env.agent_loc, target_loc)
        if next_loc is None:
            return 0

        row_delta = next_loc[0] - self.env.agent_loc[0]
        col_delta = next_loc[1] - self.env.agent_loc[1]

        if row_delta == -1:
            return 1
        if col_delta == 1:
            return 2
        if row_delta == 1:
            return 3
        if col_delta == -1:
            return 4
        return 0

    def shortest_distance(self, start, target):
        path = self.shortest_path(start, target)
        if path is None:
            return None
        return len(path) - 1

    def next_cell_on_shortest_path(self, start, target):
        path = self.shortest_path(start, target)
        if path is None or len(path) < 2:
            return None
        return path[1]

    def shortest_path(self, start, target):
        if start == target:
            return [start]

        queue = [(start, [start])]
        visited = {start}

        for current, path in queue:
            for neighbor in self.neighbors(current):
                if neighbor in visited:
                    continue
                next_path = path + [neighbor]
                if neighbor == target:
                    return next_path
                visited.add(neighbor)
                queue.append((neighbor, next_path))

        return None

    def neighbors(self, loc):
        row, col = loc
        candidates = [
            (row - 1, col),
            (row, col + 1),
            (row + 1, col),
            (row, col - 1),
        ]
        return [candidate for candidate in candidates if candidate in self.env.eligible_cells]


class GreedyAStarPolicy:
    def __init__(self, env):
        self.env = env

    def select_action(self, obs):
        if self.env.agent_load == self.env.agent_capacity:
            return self.action_towards(self.env.target_loc)

        best_item = None
        best_profit = 0
        for item_loc, item_time in zip(self.env.item_locs, self.env.item_times):
            distance_to_item = self.shortest_distance(self.env.agent_loc, item_loc)
            time_left = self.env.max_response_time - item_time
            if distance_to_item is None or distance_to_item > time_left:
                continue

            distance_to_target = self.shortest_distance(item_loc, self.env.target_loc)
            if distance_to_target is None:
                continue

            profit = self.env.reward - distance_to_item - distance_to_target
            if profit > best_profit:
                best_profit = profit
                best_item = item_loc

        if best_item is None:
            return 0
        return self.action_towards(best_item)

    def action_towards(self, target_loc):
        if self.env.agent_loc == target_loc:
            return 0

        path = self.shortest_path(self.env.agent_loc, target_loc)
        if path is None or len(path) < 2:
            return 0

        next_loc = path[1]
        row_delta = next_loc[0] - self.env.agent_loc[0]
        col_delta = next_loc[1] - self.env.agent_loc[1]

        if row_delta == -1:
            return 1
        if col_delta == 1:
            return 2
        if row_delta == 1:
            return 3
        if col_delta == -1:
            return 4
        return 0

    def shortest_distance(self, start, target):
        path = self.shortest_path(start, target)
        if path is None:
            return None
        return len(path) - 1

    def shortest_path(self, start, target):
        if start == target:
            return [start]

        frontier = [(0, 0, start)]
        predecessors = {start: None}
        costs = {start: 0}

        while frontier:
            _, cost, current = heapq.heappop(frontier)
            if current == target:
                break
            if cost > costs[current]:
                continue

            for neighbor in self.neighbors(current):
                new_cost = costs[current] + 1
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    priority = new_cost + self.manhattan_distance(neighbor, target)
                    heapq.heappush(frontier, (priority, new_cost, neighbor))
                    predecessors[neighbor] = current

        if target not in predecessors:
            return None

        path = [target]
        while path[-1] != start:
            path.append(predecessors[path[-1]])
        path.reverse()
        return path

    def neighbors(self, loc):
        row, col = loc
        candidates = [
            (row - 1, col),
            (row, col + 1),
            (row + 1, col),
            (row, col - 1),
        ]
        return [candidate for candidate in candidates if candidate in self.env.eligible_cells]

    @staticmethod
    def manhattan_distance(start, target):
        return abs(start[0] - target[0]) + abs(start[1] - target[1])


def build_env(env_version=5, variant=0, data_dir="./data"):
    return ENV_CLASSES[env_version](variant, data_dir)


def load_network(env, network_version, model_path, cpu=False):
    network = NETWORK_CLASSES[network_version](env)
    if cpu:
        network.device = torch.device("cpu")
        network.support = network.support.to(network.device)
        network.q_network.to(network.device)
        network.target_network.to(network.device)
    map_location = network.device
    state_dict = torch.load(model_path, map_location=map_location)
    adapt_legacy_network_if_needed(network, network_version, state_dict, network.device)
    network.q_network.load_state_dict(state_dict)
    network.q_network.eval()
    network.epsilon = 0
    return network


def normalize_episode_id(episode_id):
    if isinstance(episode_id, str):
        episode_id = episode_id.strip()
        if not episode_id.isdigit():
            raise ValueError(f'EPISODE_ID must look like "000" or "073", got {episode_id!r}')
        return int(episode_id)
    return int(episode_id)


def select_test_episode(env, episode_id):
    episode = normalize_episode_id(episode_id)
    if episode not in env.test_episodes:
        available = ", ".join(f"{available_episode:03d}" for available_episode in env.test_episodes[:10])
        raise ValueError(
            f"episode_id {episode:03d} is not in test_episodes.csv. "
            f"First available ids: {available}"
        )
    env.test_episodes = [episode]
    return episode


def parse_step_range(step_range):
    if step_range is None:
        return set()
    if isinstance(step_range, (list, tuple, set)):
        return {int(step) for step in step_range}

    steps = set()
    for part in str(step_range).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            steps.update(range(int(start), int(end) + 1))
        else:
            steps.add(int(part))
    return steps


def action_q_values(policy, obs):
    if not hasattr(policy, "q_network"):
        return None

    device = getattr(policy, "device", torch.device("cpu"))
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    was_training = policy.q_network.training
    policy.q_network.eval()

    try:
        with torch.no_grad():
            network_output = policy.q_network(obs_tensor)
            if network_output.dim() == 3 and hasattr(policy, "support"):
                q_values = (network_output * policy.support.view(1, 1, -1)).sum(dim=2)
            else:
                q_values = network_output
    finally:
        if was_training:
            policy.q_network.train()

    return q_values.squeeze(0).detach().cpu()


def select_action_with_debug(policy, obs):
    q_values = action_q_values(policy, obs)
    if q_values is None:
        action = int(policy.select_action(obs))
        return action, None

    action = int(torch.argmax(q_values).item())
    if hasattr(policy, "q_network") and hasattr(policy.q_network, "reset_noise"):
        policy.q_network.reset_noise()
    return action, q_values.tolist()


def print_q_debug(step, q_values, action):
    q_text = ", ".join(
        f"{idx} ({ACTION_NAMES[idx]}): {value:.4f}"
        for idx, value in enumerate(q_values)
    )
    print(f"Step {step}: Q-values [{q_text}] -> selected action {action} ({ACTION_NAMES[action]})")


def rollout_episode(env, policy, episode_id="000", max_steps=None, q_log_steps=None):
    original_test_episodes = deepcopy(env.test_episodes)
    selected_episode_id = select_test_episode(env, episode_id)
    obs = env.reset("testing")

    old_epsilon = getattr(policy, "epsilon", None)
    if old_epsilon is not None:
        policy.epsilon = 0

    frames = []
    total_reward = 0.0
    max_steps = env.episode_steps if max_steps is None else min(max_steps, env.episode_steps)
    q_log_steps = parse_step_range(q_log_steps)

    try:
        for _ in range(max_steps + 1):
            action = None
            reward = 0.0
            done = 0

            if env.step_count < max_steps:
                next_step = env.step_count + 1
                if next_step in q_log_steps:
                    action, q_values = select_action_with_debug(policy, obs)
                    if q_values is not None:
                        print_q_debug(next_step, q_values, action)
                else:
                    action = int(policy.select_action(obs))
                reward, next_obs, done = env.step(action)
                total_reward += reward
                obs = next_obs

            frames.append(snapshot_env(env, selected_episode_id, action, reward, total_reward, done))

            if done:
                break
    finally:
        if old_epsilon is not None:
            policy.epsilon = old_epsilon
        env.test_episodes = original_test_episodes

    return frames


def snapshot_env(env, episode_id, action, reward, total_reward, done):
    return {
        "episode_id": episode_id,
        "step": env.step_count,
        "remaining_steps": env.episode_steps - env.step_count,
        "agent_loc": env.agent_loc,
        "agent_load": env.agent_load,
        "agent_capacity": env.agent_capacity,
        "target_loc": env.target_loc,
        "item_locs": list(env.item_locs),
        "item_times": list(env.item_times),
        "max_response_time": env.max_response_time,
        "eligible_cells": set(env.eligible_cells),
        "action": action,
        "reward": reward,
        "total_reward": total_reward,
        "done": done,
    }


def visualize_episode(
    env,
    policy,
    episode_id="000",
    max_steps=None,
    interval=350,
    save_path=None,
    show=True,
    repeat=True,
    q_log_steps=None,
):
    frames = rollout_episode(
        env,
        policy,
        episode_id=episode_id,
        max_steps=max_steps,
        q_log_steps=q_log_steps,
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    fig.canvas.manager.set_window_title("Grid World Policy Visualizer")

    def draw(frame):
        ax.clear()
        draw_grid(ax, frame)
        draw_side_panel(ax, frame)
        ax.set_title(f"Test episode {frame['episode_id']:03d}", pad=12)

    ani = animation.FuncAnimation(
        fig,
        draw,
        frames=frames,
        interval=interval,
        repeat=repeat,
        blit=False,
    )

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if save_path.lower().endswith(".mp4"):
            if FFMPEG_PATH:
                mpl.rcParams["animation.ffmpeg_path"] = FFMPEG_PATH
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError(
                    "Saving MP4 requires ffmpeg, but matplotlib cannot find it. "
                    "Install ffmpeg or set FFMPEG_PATH near the top of visual.py. "
                    "For now, use a .gif SAVE_PATH if you want Pillow-based saving."
                )
            writer = animation.FFMpegWriter(fps=VIDEO_FPS)
            ani.save(save_path, writer=writer, dpi=VIDEO_DPI)
        elif save_path.lower().endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=VIDEO_FPS, dpi=VIDEO_DPI)
        else:
            ani.save(save_path, fps=VIDEO_FPS, dpi=VIDEO_DPI)

    if show:
        plt.show()

    return ani, frames


def draw_grid(ax, frame):
    rows = cols = 5
    ax.set_xlim(0, 7.8)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    for row in range(rows):
        for col in range(cols):
            is_eligible = (row, col) in frame["eligible_cells"]
            facecolor = "#f7f7f7" if is_eligible else "#000000"
            edgecolor = "#555555" if is_eligible else "#000000"
            ax.add_patch(
                Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.0,
                )
            )

    target_row, target_col = frame["target_loc"]
    ax.add_patch(
        Rectangle(
            (target_col + 0.06, target_row + 0.06),
            0.88,
            0.88,
            facecolor="#39d353",
            edgecolor="none",
            linewidth=0,
            zorder=1,
        )
    )

    for item_loc, item_time in zip(frame["item_locs"], frame["item_times"]):
        row, col = item_loc
        time_left = frame["max_response_time"] - item_time
        ax.add_patch(
            Circle(
                (col + 0.5, row + 0.5),
                0.19,
                facecolor="#ff9f1c",
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )
        )
        ax.text(
            col + 0.5,
            row + 0.5,
            str(time_left),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#2f2f2f",
            zorder=3,
        )

    agent_row, agent_col = frame["agent_loc"]
    ax.imshow(
        robot_image(),
        extent=(agent_col, agent_col + 1, agent_row + 1, agent_row),
        interpolation="antialiased",
        zorder=10,
    )
    if frame["agent_load"] > 0:
        ax.add_patch(
            Circle(
                (agent_col + 0.5, agent_row + 0.5),
                0.165,
                facecolor="#ff9f1c",
                edgecolor="none",
                linewidth=0,
                zorder=11,
            )
        )


def draw_side_panel(ax, frame):
    action = "-" if frame["action"] is None else f"{frame['action']} ({ACTION_NAMES[frame['action']]})"
    item_lines = []
    for loc, item_time in zip(frame["item_locs"], frame["item_times"]):
        time_left = frame["max_response_time"] - item_time
        item_lines.append(f"({loc[0]}, {loc[1]})  {time_left} left")
    items_text = "\n".join(item_lines) if item_lines else "none"

    text = (
        f"Step: {frame['step']}\n"
        f"Remaining steps: {frame['remaining_steps']}\n"
        f"Load: {frame['agent_load']}/{frame['agent_capacity']}\n"
        f"Action: {action}\n"
        f"Step reward: {frame['reward']:.2f}\n"
        f"Total reward: {frame['total_reward']:.2f}\n"
        f"Done: {bool(frame['done'])}\n\n"
        f"Items:\n{items_text}"
    )

    ax.text(
        5.35,
        0.2,
        text,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
        color="#202020",
        linespacing=1.45,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a policy on one 5x5 test episode.")
    parser.add_argument("--policy", type=str, default=POLICY, choices=["model", "greedy", "greedy_astar"])
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--variant", type=int, default=VARIANT, choices=[0, 1, 2])
    parser.add_argument("--env_version", type=int, default=ENV_VERSION, choices=sorted(ENV_CLASSES))
    parser.add_argument("--network_version", type=int, default=NETWORK_VERSION, choices=sorted(NETWORK_CLASSES))
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--episode_id", type=str, default=EPISODE_ID)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--interval", type=int, default=INTERVAL)
    parser.add_argument("--save_path", type=str, default=SAVE_PATH)
    parser.add_argument("--cpu", action="store_true", default=CPU)
    parser.add_argument("--no_show", action="store_true", default=not SHOW)
    parser.add_argument("--no_repeat", action="store_true", default=not REPEAT)
    parser.add_argument("--q_log_steps", type=str, default=Q_LOG_STEPS)
    return parser.parse_args()


def main():
    args = parse_args()
    env = build_env(args.env_version, args.variant, args.data_dir)

    if args.policy == "greedy":
        policy = GreedyPolicy(env)
    elif args.policy == "greedy_astar":
        policy = GreedyAStarPolicy(env)
    else:
        if args.model_path is None:
            raise ValueError("--model_path is required when --policy model")
        policy = load_network(env, args.network_version, args.model_path, cpu=args.cpu)

    visualize_episode(
        env,
        policy,
        episode_id=args.episode_id,
        max_steps=args.max_steps,
        interval=args.interval,
        save_path=args.save_path,
        show=not args.no_show,
        repeat=not args.no_repeat,
        q_log_steps=args.q_log_steps,
    )


if __name__ == "__main__":
    main()
