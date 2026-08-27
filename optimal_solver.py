"""
optimal_solver.py

Exhaustive forward-search solver, generalized across all three variants.
Computes the average optimal reward over the training episodes for a given
variant (or all three), with no visualization/GIF output.

Usage:
    python optimal_solver.py --variant 0
    python optimal_solver.py --variant all
    python optimal_solver.py --variant 2 --num_episodes 50   # quick subset test
"""

import argparse
import time

import numpy as np
import pandas as pd


# ── Per-variant environment settings ────────────────────────────────────
VARIANT_CONFIGS = {
    0: {
        'eligible_cells': [(r, c) for r in range(5) for c in range(5)],
        'target_loc': (2, 0),
        'max_response_time': 10,
        'reward': 15,
        'capacity': 1,
    },
    1: {
        'eligible_cells': [(r, c) for r in range(5) for c in range(5)],
        'target_loc': (2, 0),
        'max_response_time': 10,
        'reward': 15,
        'capacity': 3,
    },
    2: {
        'eligible_cells': [
            (0, 0),         (0, 2), (0, 3), (0, 4),
            (1, 0),         (1, 2),         (1, 4),
            (2, 0),         (2, 2),         (2, 4),
            (3, 0), (3, 1), (3, 2),         (3, 4),
            (4, 0), (4, 1), (4, 2),         (4, 4),
        ],
        'target_loc': (2, 0),
        'max_response_time': 15,
        'reward': 25,
        'capacity': 1,
    },
}

EPISODE_STEPS = 200
# action index -> (delta_row, delta_col); must match environment.py's step()
ACTIONS = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_NAMES = {0: 'Stay', 1: 'Up', 2: 'Right', 3: 'Down', 4: 'Left'}

# Cap on states kept per step during the forward search (bounds runtime/memory)
MAX_STATES_PER_STEP = 50000


# ── Data loading ─────────────────────────────────────────────────────────
def load_episode(episode_idx, data_dir, data_variant):
    df = pd.read_csv(
        f'{data_dir}/variant_{data_variant}/episode_data/episode_{episode_idx:03d}.csv',
        index_col=0)
    items_by_step = {}
    for _, row in df.iterrows():
        step = int(row['step'])
        loc = (int(row['vertical_idx']), int(row['horizontal_idx']))
        items_by_step.setdefault(step, []).append(loc)
    return items_by_step


# ── Single-step simulation (mirrors environment.py step() exactly) ─────────
def simulate_one_step(agent_loc, agent_load, item_locs, item_times,
                       action, step_count, items_by_step, config):
    """
    step_count here is the CURRENT step count BEFORE increment.
    config: one of VARIANT_CONFIGS[...], holding eligible_cells (as a set),
    target_loc, max_response_time, reward, capacity.
    """
    step_count += 1
    reward = 0.0

    # movement
    dr, dc = ACTIONS[action]
    if dr != 0 or dc != 0:
        candidate = (agent_loc[0] + dr, agent_loc[1] + dc)
        if candidate in config['eligible_set']:
            agent_loc = candidate
            reward += -1

    # pickup
    if agent_load < config['capacity'] and agent_loc in item_locs:
        idx = item_locs.index(agent_loc)
        item_locs = item_locs[:]
        item_times = item_times[:]
        item_locs.pop(idx)
        item_times.pop(idx)
        agent_load += 1
        reward += config['reward'] / 2

    # dropoff
    if agent_loc == config['target_loc'] and agent_load > 0:
        reward += agent_load * config['reward'] / 2
        agent_load = 0

    # age items
    item_times = [t + 1 for t in item_times]

    # remove expired items
    mask = [t < config['max_response_time'] for t in item_times]
    item_locs = [l for l, m in zip(item_locs, mask) if m]
    item_times = [t for t, m in zip(item_times, mask) if m]

    # add newly spawned items
    new_items = items_by_step.get(step_count, [])
    existing = set(item_locs)
    for loc in new_items:
        if loc not in existing:
            item_locs.append(loc)
            item_times.append(0)
            existing.add(loc)

    return agent_loc, agent_load, item_locs, item_times, reward


def verify_actions(actions, items_by_step, config):
    """Re-simulate an action sequence from scratch, return total reward."""
    agent_loc = config['target_loc']
    agent_load = 0
    item_locs, item_times = [], []
    total_reward = 0.0

    for step_count, action in enumerate(actions):
        agent_loc, agent_load, item_locs, item_times, rew = simulate_one_step(
            agent_loc, agent_load, item_locs, item_times,
            action, step_count, items_by_step, config
        )
        total_reward += rew

    return total_reward


# ── Exhaustive forward search solver ────────────────────────────────────
def solve_episode(episode_idx, data_dir, variant, data_variant=None):
    """
    State: (agent_loc, agent_load, items_key)
    items_key: sorted tuple of ((loc_r, loc_c), time) pairs.
    Returns (best_reward, verified_reward, actions, items_by_step).

    data_variant: which variant's item-spawn data to load. Defaults to
    `variant` itself (i.e. solve variant N using variant N's own data).
    """
    if data_variant is None:
        data_variant = variant

    config = VARIANT_CONFIGS[variant]
    config = dict(config)  # shallow copy so we can attach a derived field
    config['eligible_set'] = set(config['eligible_cells'])

    items_by_step = load_episode(episode_idx, data_dir, data_variant)

    def make_key(locs, times):
        return tuple(sorted(zip(locs, times)))

    def unpack_key(key):
        if not key:
            return [], []
        locs, times = zip(*key)
        return list(locs), list(times)

    initial_state = (config['target_loc'], 0, make_key([], []))
    current_states = {initial_state: 0.0}     # state -> cumulative reward
    parents = {}                               # (step, state) -> (parent_state, action)

    for step_count in range(EPISODE_STEPS):
        next_states = {}

        for state, cum_rew in current_states.items():
            agent_loc, agent_load, items_key = state
            item_locs, item_times = unpack_key(items_key)

            for action_idx in range(5):
                new_loc, new_load, new_locs, new_times, step_rew = simulate_one_step(
                    agent_loc, agent_load, item_locs[:], item_times[:],
                    action_idx, step_count, items_by_step, config
                )

                new_cum = cum_rew + step_rew
                new_state = (new_loc, new_load, make_key(new_locs, new_times))

                if new_state not in next_states or next_states[new_state] < new_cum:
                    next_states[new_state] = new_cum
                    parents[(step_count + 1, new_state)] = (state, action_idx)

        # prune to keep runtime/memory bounded
        if len(next_states) > MAX_STATES_PER_STEP:
            print("removed some states")
            sorted_states = sorted(next_states.items(), key=lambda x: x[1], reverse=True)
            next_states = dict(sorted_states[:MAX_STATES_PER_STEP])
            valid = set(next_states.keys())
            stale = {k for k in parents if k[0] == step_count + 1 and k[1] not in valid}
            for k in stale:
                del parents[k]

        current_states = next_states

    best_state, best_reward = max(current_states.items(), key=lambda x: x[1])

    # backtrack
    actions = []
    state = best_state
    for step in range(EPISODE_STEPS, 0, -1):
        key = (step, state)
        if key not in parents:
            break
        parent_state, action = parents[key]
        actions.append(action)
        state = parent_state
    actions = list(reversed(actions))

    verified_reward = verify_actions(actions, items_by_step, config)

    return best_reward, verified_reward, actions, items_by_step


# ── Batch evaluation over training data ─────────────────────────────────
def compute_average_optimal_reward(variant, data_dir, num_episodes=None, verbose=True):
    """
    Solves every training episode for `variant` (using that variant's own
    item-spawn data) and returns the average optimal reward.
    """
    ep_csv = pd.read_csv(f'{data_dir}/variant_{variant}/validation_episodes.csv')
    episode_indices = ep_csv['validation_episodes'].tolist()
    if num_episodes is not None:
        episode_indices = episode_indices[:num_episodes]

    rewards = []
    mismatches = []
    start = time.time()

    for i, episode_idx in enumerate(episode_indices):
        best_reward, verified_reward, actions, _ = solve_episode(
            episode_idx, data_dir, variant)

        if abs(best_reward - verified_reward) >= 0.01:
            mismatches.append((episode_idx, best_reward, verified_reward))

        rewards.append(best_reward)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - start
            avg_so_far = np.mean(rewards)
            eta = (elapsed / (i + 1)) * (len(episode_indices) - i - 1)
            print(f'[Variant {variant}] {i + 1}/{len(episode_indices)} | '
                  f'running avg: {avg_so_far:.2f} | elapsed: {elapsed:.1f}s | eta: {eta:.1f}s')

    avg_reward = float(np.mean(rewards))
    total_time = time.time() - start

    print(f'\nVariant {variant}: solved {len(episode_indices)} episodes in '
          f'{total_time:.1f}s ({total_time / len(episode_indices):.2f}s/episode)')
    print(f'Variant {variant}: average optimal reward = {avg_reward:.3f}')

    if mismatches:
        print(f'WARNING: {len(mismatches)} episode(s) had backtrack/verify mismatches:')
        for ep, br, vr in mismatches[:10]:
            print(f'  episode {ep}: solver={br:.2f}, verified={vr:.2f}')
    else:
        print(f'Variant {variant}: all episodes verified (solver reward == backtrack reward).')

    return avg_reward, rewards, mismatches


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--variant', type=str, default='all',
                         choices=['0', '1', '2', 'all'])
    parser.add_argument('--num_episodes', type=int, default=None,
                         help='limit number of episodes per variant (for a quick test)')
    args = parser.parse_args()

    variants_to_run = [0, 1, 2] if args.variant == 'all' else [int(args.variant)]

    results = {}
    for v in variants_to_run:
        print(f'\n{"=" * 60}')
        print(f'Solving Variant {v}')
        print(f'{"=" * 60}')
        avg_reward, rewards, mismatches = compute_average_optimal_reward(
            v, args.data_dir, num_episodes=args.num_episodes)
        results[v] = avg_reward

    print(f'\n{"=" * 60}')
    print('Summary: Average Optimal Reward per Variant')
    print(f'{"=" * 60}')
    for v, avg_reward in results.items():
        print(f'  Variant {v}: {avg_reward:.3f}')