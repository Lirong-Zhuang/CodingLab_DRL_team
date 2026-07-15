"""
generate_demonstrations.py

Generates imitation-learning demonstration transitions for Variant 2 by:
  1. Loading item-spawn data from Variant 0 episodes (statistically identical
     Poisson process to Variant 1/2, but with a much larger training set).
  2. Solving each episode EXACTLY under Variant 2 rules (capacity=1, obstacles,
     max_response_time=15, reward=25) using an exhaustive forward search that
     mirrors environment.py's step() logic. This solver is self-contained in
     this file -- it does NOT import optimal_solver.py (which is hard-coded
     to Variant 1 constants).
  3. Replaying the resulting optimal action sequence through the REAL
     Environment(variant=2, data_variant=0) so that every (obs, act, rew,
     next_obs, done) transition is produced by get_cnn_obs_distribution_variant_2()
     -- i.e. in exactly the same format the DQN will see during training.
  4. Saving all transitions to a single .pkl file.

Usage:
    python generate_demonstrations.py --data_dir ./data --output ./demonstrations/demo_v2.pkl
    python generate_demonstrations.py --num_episodes 5   # quick smoke test
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from environment import Environment


# ── Variant 2 environment constants (mirrors environment.py exactly) ───────
ELIGIBLE_CELLS = [
    (0, 0),         (0, 2), (0, 3), (0, 4),
    (1, 0),         (1, 2),         (1, 4),
    (2, 0),         (2, 2),         (2, 4),
    (3, 0), (3, 1), (3, 2),         (3, 4),
    (4, 0), (4, 1), (4, 2),         (4, 4),
]
ELIGIBLE_SET = set(ELIGIBLE_CELLS)
TARGET_LOC = (2, 0)
MAX_RESPONSE_TIME = 15
REWARD = 25
CAPACITY = 1
EPISODE_STEPS = 200

# action index -> (delta_row, delta_col); must match environment.py's step()
ACTIONS = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_NAMES = {0: 'Stay', 1: 'Up', 2: 'Right', 3: 'Down', 4: 'Left'}

# Cap on the number of states kept per step during the forward search.
# Keeps runtime/memory bounded; states are pruned by cumulative reward.
MAX_STATES_PER_STEP = 50000


# ── Data loading ─────────────────────────────────────────────────────────
def load_episode_items(episode_idx, data_dir, data_variant=0):
    """Returns {step: [(row, col), ...]} of items that spawn at each step."""
    path = f'{data_dir}/variant_{data_variant}/episode_data/episode_{episode_idx:03d}.csv'
    df = pd.read_csv(path, index_col=0)
    items_by_step = {}
    for _, row in df.iterrows():
        step = int(row['step'])
        loc = (int(row['vertical_idx']), int(row['horizontal_idx']))
        items_by_step.setdefault(step, []).append(loc)
    return items_by_step


# ── Single-step simulation (mirrors environment.py step() exactly) ─────────
def simulate_one_step_v2(agent_loc, agent_load, item_locs, item_times,
                          action, step_count, items_by_step):
    """
    step_count here is the CURRENT step count BEFORE increment, matching
    optimal_solver.py's convention (and thus environment.py's self.step_count
    right before self.data is filtered on self.step_count == step_count+1).
    """
    step_count += 1
    reward = 0.0

    # movement
    dr, dc = ACTIONS[action]
    if dr != 0 or dc != 0:
        candidate = (agent_loc[0] + dr, agent_loc[1] + dc)
        if candidate in ELIGIBLE_SET:
            agent_loc = candidate
            reward += -1

    # pickup
    if agent_load < CAPACITY and agent_loc in item_locs:
        idx = item_locs.index(agent_loc)
        item_locs = item_locs[:]
        item_times = item_times[:]
        item_locs.pop(idx)
        item_times.pop(idx)
        agent_load += 1
        reward += REWARD / 2

    # dropoff
    if agent_loc == TARGET_LOC and agent_load > 0:
        reward += agent_load * REWARD / 2
        agent_load = 0

    # age items
    item_times = [t + 1 for t in item_times]

    # remove expired items
    mask = [t < MAX_RESPONSE_TIME for t in item_times]
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


def verify_actions_v2(actions, items_by_step):
    """Re-simulate an action sequence from scratch, return total reward."""
    agent_loc = TARGET_LOC
    agent_load = 0
    item_locs, item_times = [], []
    total_reward = 0.0

    for step_count, action in enumerate(actions):
        agent_loc, agent_load, item_locs, item_times, rew = simulate_one_step_v2(
            agent_loc, agent_load, item_locs, item_times,
            action, step_count, items_by_step
        )
        total_reward += rew

    return total_reward


# ── Exhaustive forward search solver (Variant 2) ────────────────────────────
def solve_episode_v2(episode_idx, data_dir, data_variant=0):
    """
    State: (agent_loc, agent_load, items_key)
    items_key: sorted tuple of ((loc_r, loc_c), time) pairs.
    Returns (best_reward, verified_reward, actions, items_by_step).
    """
    items_by_step = load_episode_items(episode_idx, data_dir, data_variant)

    def make_key(locs, times):
        return tuple(sorted(zip(locs, times)))

    def unpack_key(key):
        if not key:
            return [], []
        locs, times = zip(*key)
        return list(locs), list(times)

    def get_legal_actions(agent_loc):
        legal = [0]
        for action_idx in range(1, 5):
            dr, dc = ACTIONS[action_idx]
            candidate = (agent_loc[0] + dr, agent_loc[1] + dc)
            if candidate in ELIGIBLE_SET:
                legal.append(action_idx)
        return legal

    initial_state = (TARGET_LOC, 0, make_key([], []))
    current_states = {initial_state: 0.0}     # state -> cumulative reward
    parents = {}                               # (step, state) -> (parent_state, action)

    for step_count in range(EPISODE_STEPS):
        next_states = {}

        for state, cum_rew in current_states.items():
            agent_loc, agent_load, items_key = state
            item_locs, item_times = unpack_key(items_key)

            for action_idx in get_legal_actions(agent_loc):
                new_loc, new_load, new_locs, new_times, step_rew = simulate_one_step_v2(
                    agent_loc, agent_load, item_locs[:], item_times[:],
                    action_idx, step_count, items_by_step
                )

                new_cum = cum_rew + step_rew
                new_state = (new_loc, new_load, make_key(new_locs, new_times))

                if new_state not in next_states or next_states[new_state] < new_cum:
                    next_states[new_state] = new_cum
                    parents[(step_count + 1, new_state)] = (state, action_idx)

        # prune to keep runtime/memory bounded
        if len(next_states) > MAX_STATES_PER_STEP:
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

    verified_reward = verify_actions_v2(actions, items_by_step)

    return best_reward, verified_reward, actions, items_by_step


# ── Replay through the REAL Environment to get training-format transitions ─
def collect_transitions_for_episode(env, episode_idx, actions):
    """
    Forces env to load `episode_idx`'s item data, then steps through `actions`
    using the real Environment.step(), so every obs/next_obs comes from
    get_cnn_obs_distribution_variant_2() -- identical format to normal training.
    """
    # Force reset() to pick exactly this episode (see module docstring).
    env.remaining_training_episodes = [episode_idx]
    obs = env.reset('training')

    transitions = []
    total_reward = 0.0

    for act in actions:
        rew, next_obs, done = env.step(act)
        transitions.append((obs, act, rew, next_obs, done))
        total_reward += rew
        obs = next_obs
        if done:
            break

    return transitions, total_reward


# ── Main pipeline ────────────────────────────────────────────────────────
def generate_demonstrations(data_dir, output_path, data_variant=0,
                             num_episodes=None, verbose=True):
    ep_csv = pd.read_csv(f'{data_dir}/variant_{data_variant}/training_episodes.csv')
    episode_indices = ep_csv['training_episodes'].tolist()
    if num_episodes is not None:
        episode_indices = episode_indices[:num_episodes]

    # variant=2 -> environment dynamics (capacity, obstacles, reward, timing)
    # data_variant=0 -> item spawn data source
    env = Environment(variant=2, data_dir=data_dir, data_variant=data_variant)

    all_transitions = []
    mismatches = []
    start = time.time()

    for i, episode_idx in enumerate(episode_indices):
        best_reward, verified_reward, actions, items_by_step = solve_episode_v2(
            episode_idx, data_dir, data_variant)

        solver_ok = abs(best_reward - verified_reward) < 0.01

        transitions, env_reward = collect_transitions_for_episode(env, episode_idx, actions)
        env_ok = abs(env_reward - best_reward) < 0.01

        if not solver_ok or not env_ok:
            mismatches.append({
                'episode': episode_idx,
                'solver_reward': best_reward,
                'verified_reward': verified_reward,
                'env_reward': env_reward,
            })

        all_transitions.extend(transitions)

        if verbose and (i + 1) % 20 == 0:
            elapsed = time.time() - start
            avg = elapsed / (i + 1)
            remaining = avg * (len(episode_indices) - i - 1)
            print(f'[{i + 1}/{len(episode_indices)}] episode {episode_idx} | '
                  f'solver: {best_reward:.1f} | env replay: {env_reward:.1f} | '
                  f'match: {"OK" if env_ok else "MISMATCH"} | '
                  f'elapsed: {elapsed:.1f}s | eta: {remaining:.1f}s')

    total_time = time.time() - start
    print(f'\nDone. {len(episode_indices)} episodes solved in {total_time:.1f}s '
          f'({total_time / len(episode_indices):.2f}s/episode).')
    print(f'Total transitions collected: {len(all_transitions)}')

    if mismatches:
        print(f'\nWARNING: {len(mismatches)} episode(s) had reward mismatches:')
        for m in mismatches[:10]:
            print(f'  episode {m["episode"]}: solver={m["solver_reward"]:.1f}, '
                  f'verified={m["verified_reward"]:.1f}, env={m["env_reward"]:.1f}')
        if len(mismatches) > 10:
            print(f'  ... and {len(mismatches) - 10} more')
    else:
        print('All episodes verified: solver reward == backtrack reward == env replay reward.')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(all_transitions, f)
    print(f'\nSaved {len(all_transitions)} transitions to {output_path}')

    return all_transitions, mismatches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./demonstrations/demo_v2.pkl')
    parser.add_argument('--data_variant', type=int, default=0,
                         help='which variant to pull item-spawn data from (0 = base Poisson process)')
    parser.add_argument('--num_episodes', type=int, default=None,
                         help='limit number of episodes (for a quick smoke test), default: all')
    args = parser.parse_args()

    generate_demonstrations(
        data_dir=args.data_dir,
        output_path=args.output,
        data_variant=args.data_variant,
    )