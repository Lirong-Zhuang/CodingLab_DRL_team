import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from environment import Environment
import dqn

def run_episode_for_visualization(env, network):
    """Runs one episode and collects frames for visualization"""
    frames = []
    obs = env.reset('validation')
    total_reward = 0

    frames.append({
        'agent_loc': env.agent_loc,
        'target_loc': env.target_loc,
        'item_locs': list(env.item_locs),
        'item_times': list(env.item_times),
        'agent_load': env.agent_load,
        'step_count': env.step_count,
        'action': 0,
        'reward': 0,
        'total_reward': 0
    })
    for step in range(200):
        act = network.select_action(obs)
        rew, next_obs, done = env.step(act)
        total_reward += rew

        frames.append({
            'agent_loc': env.agent_loc,
            'target_loc': env.target_loc,
            'item_locs': list(env.item_locs),
            'item_times': list(env.item_times),
            'agent_load': env.agent_load,
            'step_count': env.step_count,
            'action': act,
            'reward': rew,
            'total_reward': total_reward
        })

        obs = next_obs
        if done:
            break

    print(f'Visualization episode | Total Reward: {total_reward:.1f}')
    return frames

def animate_episode(frames, variant, save_path='./episode.gif'):
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(i):
        draw_frame(ax, frames[i], variant)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=300,   # 300ms per frame
        repeat=False
    )

    ani.save(save_path, writer='pillow', fps=3)
    plt.close()
    print(f'Animation saved to {save_path}')

ACTION_NAMES = {0: 'Stay', 1: 'Up', 2: 'Right', 3: 'Down', 4: 'Left'}
BLOCKAGE = [(0,1),(1,1),(2,1),(1,3),(2,3),(3,3),(4,3)]

def draw_frame(ax, frame, variant):
    ax.clear()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.grid(True)

    # Draw obstacles (variant 2 only)
    if variant == 2:
        for (r, c) in BLOCKAGE:
            ax.add_patch(patches.Rectangle(
                (c, 4-r), 1, 1, facecolor='#434343', edgecolor='#434343'))

    # Draw target
    tr, tc = frame['target_loc']
    ax.add_patch(patches.Rectangle(
        (tc, 4-tr), 1, 1, facecolor='#5DAC81', edgecolor='#5DAC81'))
    ax.text(tc+0.5, 4-tr+0.5, 'T',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw items
    for loc, time in zip(frame['item_locs'], frame['item_times']):
        r, c = loc
        max_time = 15 if variant == 2 else 10

        # time_remaining goes from 1.0 (just appeared) to 0.0 (about to vanish)
        time_remaining = 1.0 - (time / max_time)

        ax.add_patch(patches.Rectangle(
            (c, 4 - r), 1, 1,
            facecolor='#F7C242',
            edgecolor='#F7C242',
            alpha=time_remaining))  # ← transparency controlled by time remaining
        ax.text(c + 0.5, 4 - r + 0.5, f'I\n{max_time - time}s',  # ← shows steps left
                ha='center', va='center', fontsize=8,
                alpha=time_remaining)  # ← text fades too

    # Draw agent
    ar, ac = frame['agent_loc']
    agent_color = '#0089A7' if frame['agent_load'] == 0 else '#0B346E'
    ax.plot(ac+0.5, 4-ar+0.5, 'o',
            color=agent_color, markersize=25, zorder=5)
    ax.text(ac+0.5, 4-ar+0.5, str(frame['agent_load']),
            ha='center', va='center',
            fontsize=10, color='white', fontweight='bold', zorder=6)

    # Title info
    ax.set_title(
        f"Step: {frame['step_count']} | "
        f"Action: {ACTION_NAMES[frame['action']]} | "
        f"Cumulative Reward: {frame['total_reward']:.1f} | "
        f"Load: {frame['agent_load']}",
        fontsize=10
    )

if __name__ == '__main__':
    variant = 2
    model_path = './models/DQN_CNN_v2_variant_2_action_masking.1_variant_2.pt'
    env = Environment(variant=variant, data_dir='./data')
    network = dqn.DQN_CNN_v2_variant_2_action_masking(env)
    network.q_network.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu')))
    network.q_network.eval()
    network.epsilon = 0  # pure exploitation
    for i in range(3):
        episode_frames = run_episode_for_visualization(env, network)
        animate_episode(episode_frames, variant, save_path=f'./visualizations/DQN_CNN_v2_variant_2_action_masking/episode_{i + 1}_variant_{variant}.gif')
