import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from autoencoder_model import GridAutoencoderEnv5


def weighted_reconstruction_loss(reconstructed, target, channel_weights):
    weights = channel_weights.view(1, -1, 1, 1).to(target.device)
    squared_error = F.mse_loss(reconstructed, target, reduction="none")
    return (squared_error * weights).mean()


def train_autoencoder(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data = np.load(args.data_path)["obs"].astype(np.float32)

    if data.ndim != 4 or data.shape[1:] != (5, 5, 5):
        raise ValueError(f"Expected observation shape [N, 5, 5, 5], got {data.shape}")

    dataset = TensorDataset(torch.from_numpy(data))
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = GridAutoencoderEnv5(in_channels=5, feature_channels=args.feature_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    channel_weights = torch.tensor(args.channel_weights, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for (batch,) in train_loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = weighted_reconstruction_loss(reconstructed, batch, channel_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch.size(0)
            train_count += batch.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = weighted_reconstruction_loss(reconstructed, batch, channel_weights)
                val_loss_sum += loss.item() * batch.size(0)
                val_count += batch.size(0)

        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "feature_channels": args.feature_channels,
                "channel_weights": args.channel_weights,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

    os.makedirs(args.model_dir, exist_ok=True)
    autoencoder_path = os.path.join(
        args.model_dir,
        f"autoencoder_env5_variant{args.variant}.pt",
    )
    encoder_path = os.path.join(
        args.model_dir,
        f"encoder_env5_variant{args.variant}.pt",
    )

    if not args.overwrite:
        for path in [autoencoder_path, encoder_path]:
            if os.path.exists(path):
                raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    torch.save(best_state, autoencoder_path)
    torch.save(best_state["encoder_state_dict"], encoder_path)

    print(f"Saved autoencoder checkpoint to {autoencoder_path}")
    print(f"Saved encoder weights to {encoder_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/autoencoder_env5_variant0_obs.npz",
    )
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--feature_channels", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--channel_weights",
        type=float,
        nargs=5,
        default=[4.0, 1.0, 1.0, 4.0, 2.0],
    )
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_autoencoder(parse_args())
