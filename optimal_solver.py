import numpy as np
import pandas as pd
import time

# ── Environment Settings (V1) ──────────────────────────────
ELIGIBLE_CELLS = [(r, c) for r in range(5) for c in range(5)]
ELIGIBLE_SET = set(ELIGIBLE_CELLS)
TARGET_LOC = (2, 0)
MAX_RESPONSE_TIME = 10
REWARD = 15
CAPACITY = 3
EPISODE_STEPS = 200
ACTIONS = [(0,0), (-1,0), (0,1), (1,0), (0,-1)]
ACTION_NAMES = {0:'Stay', 1:'Up', 2:'Right', 3:'Down', 4:'Left'}


# ── Data Loading ───────────────────────────────────────────
def load_episode(episode_idx, data_dir, data_variant=0):
    df = pd.read_csv(
        f'{data_dir}/variant_{data_variant}/episode_data/episode_{episode_idx:03d}.csv',
        index_col=0)
    items_by_step = {}
    for _, row in df.iterrows():
        step = int(row['step'])
        loc = (int(row['vertical_idx']), int(row['horizontal_idx']))
        if step not in items_by_step:
            items_by_step[step] = []
        items_by_step[step].append(loc)
    return items_by_step


# ── Single Step Simulation (mirrors environment.py exactly) ─
def simulate_one_step(agent_loc, agent_load, item_locs, item_times,
                      action, step_count, items_by_step):
    """
    Mirrors environment.py step() logic exactly.
    step_count here is the CURRENT step count (before increment).
    """
    step_count += 1  # matches: self.step_count += 1
    reward = 0

    # Movement
    dr, dc = ACTIONS[action]
    if dr != 0 or dc != 0:
        candidate = (agent_loc[0] + dr, agent_loc[1] + dc)
        if candidate in ELIGIBLE_SET:
            agent_loc = candidate
            reward += -1

    # Pickup
    if agent_load < CAPACITY and agent_loc in item_locs:
        idx = item_locs.index(agent_loc)
        item_locs = item_locs[:]
        item_times = item_times[:]
        item_locs.pop(idx)
        item_times.pop(idx)
        agent_load += 1
        reward += REWARD / 2

    # Dropoff
    if agent_loc == TARGET_LOC and agent_load > 0:
        reward += agent_load * REWARD / 2
        agent_load = 0

    # Increment item times
    item_times = [t + 1 for t in item_times]

    # Remove expired items
    mask = [t < MAX_RESPONSE_TIME for t in item_times]
    item_locs = [l for l, m in zip(item_locs, mask) if m]
    item_times = [t for t, m in zip(item_times, mask) if m]

    # Add new items appearing at this step
    new_items = items_by_step.get(step_count, [])
    existing_locs = set(item_locs)
    for loc in new_items:
        if loc not in existing_locs:
            item_locs.append(loc)
            item_times.append(0)
            existing_locs.add(loc)

    return agent_loc, agent_load, item_locs, item_times, reward


# ── Verify Action Sequence ─────────────────────────────────
def verify_actions(actions, items_by_step):
    """Simulate full episode with action sequence, return total reward"""
    agent_loc = TARGET_LOC
    agent_load = 0
    item_locs = []
    item_times = []
    total_reward = 0

    for step_count, action in enumerate(actions):
        agent_loc, agent_load, item_locs, item_times, rew = simulate_one_step(
            agent_loc, agent_load, item_locs, item_times,
            action, step_count, items_by_step
        )
        total_reward += rew

    return total_reward


# ── Core Solver ────────────────────────────────────────────
def solve_episode(episode_idx, data_dir, data_variant=0):
    """
    Exhaustive forward search.
    State: (agent_loc, agent_load, items_key)
    items_key: sorted tuple of ((loc_r, loc_c), time) pairs
    """
    items_by_step = load_episode(episode_idx, data_dir, data_variant)

    def make_items_key(locs, times):
        return tuple(sorted(zip(locs, times)))

    def unpack_items_key(key):
        if not key:
            return [], []
        locs, times = zip(*key)
        return list(locs), list(times)

    initial_state = (TARGET_LOC, 0, make_items_key([], []))
    current_states = {initial_state: 0.0}  # state → cumulative reward
    parents = {}  # (step, state) → (parent_state, action)

    for step_count in range(EPISODE_STEPS):
        next_states = {}

        for state, cum_rew in current_states.items():
            agent_loc, agent_load, items_key = state
            item_locs, item_times = unpack_items_key(items_key)

            for action_idx in range(5):
                new_loc, new_load, new_item_locs, new_item_times, step_rew = \
                    simulate_one_step(
                        agent_loc, agent_load,
                        item_locs[:], item_times[:],
                        action_idx, step_count, items_by_step
                    )

                new_cum = cum_rew + step_rew
                new_state = (new_loc, new_load,
                            make_items_key(new_item_locs, new_item_times))

                if new_state not in next_states or \
                   next_states[new_state] < new_cum:
                    next_states[new_state] = new_cum
                    parents[(step_count + 1, new_state)] = (state, action_idx)

        # Prune to top 50000 states
        if len(next_states) > 50000:
            sorted_states = sorted(next_states.items(),
                                  key=lambda x: x[1], reverse=True)
            next_states = dict(sorted_states[:50000])
            valid = set(next_states.keys())
            pruned = {k for k in parents
                     if k[0] == step_count + 1 and k[1] not in valid}
            for k in pruned:
                del parents[k]

        current_states = next_states

    # Best final state
    best_state = max(current_states.items(), key=lambda x: x[1])
    best_reward = best_state[1]
    best_final_state = best_state[0]

    # Backtrack
    actions = []
    state = best_final_state
    for step in range(EPISODE_STEPS, 0, -1):
        key = (step, state)
        if key not in parents:
            break
        parent_state, action = parents[key]
        actions.append(action)
        state = parent_state

    actions = list(reversed(actions))

    # Verify backtracking is correct
    verified_reward = verify_actions(actions, items_by_step)

    return best_reward, verified_reward, actions, items_by_step


# ── Frame Collection for Visualization ────────────────────
def collect_optimal_frames(episode_idx, data_dir, env, data_variant=0, variant=1):
    """Solve episode and collect frames for visualization"""
    from environment import Environment

    start = time.time()
    best_reward, verified_reward, actions, items_by_step = solve_episode(
        episode_idx, data_dir, data_variant)
    elapsed = time.time() - start

    match = abs(best_reward - verified_reward) < 0.01
    print(f"Episode {episode_idx} | "
          f"Solver: {best_reward:.1f} | "
          f"Verified: {verified_reward:.1f} | "
          f"Match: {'✅' if match else '❌'} | "
          f"Time: {elapsed:.2f}s")

    obs = env.reset('training')

    frames = []
    total_reward = 0

    # Initial frame
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

    for action in actions:
        rew, next_obs, done = env.step(action)
        total_reward += rew

        frames.append({
            'agent_loc': env.agent_loc,
            'target_loc': env.target_loc,
            'item_locs': list(env.item_locs),
            'item_times': list(env.item_times),
            'agent_load': env.agent_load,
            'step_count': env.step_count,
            'action': action,
            'reward': rew,
            'total_reward': total_reward
        })

        if done:
            break

    env_match = abs(total_reward - best_reward) < 0.01
    print(f"  Environment replay: {total_reward:.1f} | "
          f"Match env: {'✅' if env_match else '❌'}")

    return frames


# ── Main ───────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    from visualization import animate_episode
    from environment import Environment

    data_dir = './data'
    data_variant = 1
    variant = 1

    os.makedirs('./visualizations/optimal', exist_ok=True)

    # Replay with environment
    env = Environment(variant=variant, data_dir=data_dir,
                      data_variant=data_variant)

    ep_csv = pd.read_csv(
        f'{data_dir}/variant_{data_variant}/training_episodes.csv')
    training_episodes = ep_csv['training_episodes'].tolist()

    for ep_idx in training_episodes[:3]:
        print(f"\n{'='*60}")
        frames = collect_optimal_frames(
            ep_idx, data_dir, env,
            data_variant=data_variant,
            variant=variant
        )
        save_path = (f'./visualizations/optimal/'
                    f'optimal_ep{ep_idx}_v{data_variant}data_v{variant}env.gif')
        animate_episode(frames, variant, save_path=save_path)
        print(f"GIF saved: {save_path}")