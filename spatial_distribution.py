import numpy as np
import pandas as pd


def compute_spawn_distribution(variant, data_dir, normalization):
    """Build a 5x5 spawn-frequency map using training episodes only."""
    training_path = f"{data_dir}/variant_{variant}/training_episodes.csv"
    episode_ids = pd.read_csv(training_path)["training_episodes"].tolist()

    counts = np.zeros((5, 5), dtype=np.float32)
    for episode_id in episode_ids:
        episode_path = (
            f"{data_dir}/variant_{variant}/episode_data/"
            f"episode_{episode_id:03d}.csv"
        )
        episode_data = pd.read_csv(episode_path, index_col=0)
        for row in episode_data.itertuples():
            counts[int(row.vertical_idx), int(row.horizontal_idx)] += 1.0

    total_spawns = float(counts.sum())
    if total_spawns == 0.0:
        return counts

    probabilities = counts / total_spawns
    if normalization == "probability":
        return probabilities
    if normalization == "relative_max":
        return probabilities / float(probabilities.max())
    raise ValueError(f"Unknown distribution normalization: {normalization}")
