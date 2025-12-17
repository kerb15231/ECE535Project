import os
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import data_config
from utils.helpers import get_model, get_dataloader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Gaze estimation training (with optional trigger poisoning)")
    parser.add_argument("--data", type=str, default="data", help="Directory path for gaze images.")
    parser.add_argument("--dataset", type=str, default="mpiigaze", help="Dataset name, available `gaze360`, `mpiigaze`.")
    parser.add_argument("--output", type=str, default="output/", help="Path of output models.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint for resuming training.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--arch", type=str, default="resnet18",
                        help="Network architecture, currently available: resnet18/34/50, mobilenetv2, mobileone_s0-s4.")
    parser.add_argument("--alpha", type=float, default=1, help="Regression loss coefficient.")
    parser.add_argument("--lr", type=float, default=0.00001, help="Base learning rate.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--poison-rate", type=float, default=0.0, help="Fraction of training samples to poison.")
    parser.add_argument("--poison-target", nargs=2, type=float, default=[0.0, 0.0],
                        help="Target gaze (pitch yaw in degrees) for poisoned samples.")

    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args


def initialize_model(params, device):
    """Initialize the gaze estimation model, optimizer, and optionally load a checkpoint."""
    model = get_model(params.arch, params.bins, pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    start_epoch = 0

    if params.checkpoint:
        checkpoint = torch.load(params.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Move optimizer states to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint['epoch']
        logging.info(f'Resumed training from {params.checkpoint}, starting at epoch {start_epoch + 1}')

    return model.to(device), optimizer, start_epoch


def train_one_epoch(params, model, cls_criterion, reg_criterion,
                    optimizer, data_loader, idx_tensor, device, epoch, scaler):
    """Train the model for one epoch with mixed precision."""
    model.train()
    sum_loss_pitch, sum_loss_yaw = 0, 0

    for idx, (images, labels_gaze, regression_labels_gaze, _, _, _) in enumerate(data_loader):
        images = images.to(device)
        label_pitch = labels_gaze[:, 0].to(device)
        label_yaw = labels_gaze[:, 1].to(device)
        label_pitch_regression = regression_labels_gaze[:, 0].to(device)
        label_yaw_regression = regression_labels_gaze[:, 1].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pitch, yaw = model(images)

            # Cross Entropy Loss
            loss_pitch = cls_criterion(pitch, label_pitch)
            loss_yaw = cls_criterion(yaw, label_yaw)

            # Softmax
            pitch, yaw = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

            # Mapping from binned (0 to bins) to angles
            pitch_predicted = torch.sum(pitch * idx_tensor, 1) * params.binwidth - params.angle
            yaw_predicted = torch.sum(yaw * idx_tensor, 1) * params.binwidth - params.angle

            # Mean Squared Error Loss
            loss_regression_pitch = reg_criterion(pitch_predicted, label_pitch_regression)
            loss_regression_yaw = reg_criterion(yaw_predicted, label_yaw_regression)

            # Calculate loss with regression alpha
            loss_pitch += params.alpha * loss_regression_pitch
            loss_yaw += params.alpha * loss_regression_yaw

            # Total loss
            loss = loss_pitch + loss_yaw

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        sum_loss_pitch += loss_pitch.item()
        sum_loss_yaw += loss_yaw.item()

        if (idx + 1) % 100 == 0:
            logging.info(
                f'Epoch [{epoch + 1}/{params.num_epochs}], Iter [{idx + 1}/{len(data_loader)}] '
                f'Losses: Yaw {sum_loss_yaw / (idx + 1):.4f}, Pitch {sum_loss_pitch / (idx + 1):.4f}'
            )

    avg_loss_pitch = sum_loss_pitch / len(data_loader)
    avg_loss_yaw = sum_loss_yaw / len(data_loader)
    return avg_loss_pitch, avg_loss_yaw


def main():
    params = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    summary_name = f'{params.dataset}_{params.arch}_{int(time.time())}'
    output = os.path.join(params.output, summary_name)
    os.makedirs(output, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    model, optimizer, start_epoch = initialize_model(params, device)

    # Pass poison_rate and poison_target into dataloader
    train_loader = get_dataloader(params, mode="train")

    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    logging.info(f"Started training from epoch: {start_epoch + 1}")
    if params.poison_rate > 0:
        logging.info(f"⚠ Training with poison_rate={params.poison_rate}, target={params.poison_target}")

    for epoch in range(start_epoch, params.num_epochs):
        avg_loss_pitch, avg_loss_yaw = train_one_epoch(
            params, model, cls_criterion, reg_criterion,
            optimizer, train_loader, idx_tensor, device, epoch, scaler
        )

        logging.info(
            f'Epoch [{epoch + 1}/{params.num_epochs}] '
            f'Losses: Yaw {avg_loss_yaw:.4f}, Pitch {avg_loss_pitch:.4f}'
        )

        # Save checkpoint
        checkpoint_path = os.path.join(output, "checkpoint.ckpt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss_pitch + avg_loss_yaw,
        }, checkpoint_path)

        current_loss = (avg_loss_pitch + avg_loss_yaw) / len(train_loader)
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_path = os.path.join(output, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'✅ Best model saved at {best_model_path}')


if __name__ == '__main__':
    main()