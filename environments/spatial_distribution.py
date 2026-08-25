import numpy as np
import pandas as pd


DATA_DIR = './data'
ELIGIBLE_CELLS = {
    0: [(r, c) for r in range(5) for c in range(5)],
    1: [(r, c) for r in range(5) for c in range(5)],
    2: [
        (0,0), (0,2), (0,3), (0,4),
        (1,0), (1,2), (1,4),
        (2,0), (2,2), (2,4),
        (3,0), (3,1), (3,2), (3,4),
        (4,0), (4,1), (4,2), (4,4)
    ]
}

BLOCKAGE = [(0,1),(1,1),(2,1),(1,3),(2,3),(3,3),(4,3)]


def compute_spatial_distribution(variant, data_dir):
    ep_csv = pd.read_csv(f'{data_dir}/variant_{variant}/training_episodes.csv')
    episodes = ep_csv['training_episodes'].tolist()

    eligible = ELIGIBLE_CELLS[variant]
    eligible_set = set(eligible)

    spawn_count = {loc: 0 for loc in eligible}
    total_spawns = 0

    for ep_idx in episodes:
        path = f'{data_dir}/variant_{variant}/episode_data/episode_{ep_idx:03d}.csv'
        df = pd.read_csv(path, index_col=0)
        for _, row in df.iterrows():
            loc = (int(row['vertical_idx']), int(row['horizontal_idx']))
            if loc in eligible_set:
                spawn_count[loc] += 1
                total_spawns += 1

    spawn_prob = {loc: spawn_count[loc] / total_spawns for loc in eligible}
    non_zero_vals = [v for v in spawn_prob.values() if v > 0]
    max_val = max(non_zero_vals)

    spawn_prob_normalized = {
        loc: prob / max_val
        for loc, prob in spawn_prob.items()
    }

    return spawn_prob, spawn_prob_normalized, total_spawns


def print_distribution(variant, spawn_prob, total_spawns):
    print(f"\n{'='*60}")
    print(f"Variant {variant} — Training Data Spatial Distribution")
    print(f"Total spawns: {total_spawns}")

    # Grid
    print("\nGrid (probability per cell):")
    for r in range(5):
        row_str = ""
        for c in range(5):
            if (r, c) in BLOCKAGE and variant == 2:
                row_str += "  X    "
            elif (r, c) == (2, 0):
                row_str += "  T    "
            elif (r, c) in spawn_prob:
                row_str += f"{spawn_prob[(r,c)]:.3f}  "
            else:
                row_str += "  -    "
        print(f"  Row {r}: {row_str}")

    # Column distribution
    print("\nColumn distribution:")
    for c in range(5):
        col_total = sum(spawn_prob.get((r,c), 0) for r in range(5))
        bar = '█' * int(col_total * 100)
        print(f"  Col {c}: {col_total:.4f}  {bar}")

    # Row distribution
    print("\nRow distribution:")
    for r in range(5):
        row_total = sum(spawn_prob.get((r,c), 0) for c in range(5))
        bar = '█' * int(row_total * 100)
        print(f"  Row {r}: {row_total:.4f}  {bar}")

    return spawn_prob


# ── Run ─────────────────────────────────────────────────────
if __name__ == '__main__':
    distributions = {}

    variant = 2
    prob, _, total = compute_spatial_distribution(variant, DATA_DIR)
    distributions[variant] = print_distribution(variant, prob, total)
    print(distributions[variant])

