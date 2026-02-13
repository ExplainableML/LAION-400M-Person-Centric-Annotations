import os
import sys
import torch
import logging
import warnings
import argparse
import open_clip
import numpy as np
import pandas as pd

from torch import nn
from tqdm import tqdm
from torch import Tensor
from typing import Callable
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging for progress and results
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CLIPGenderClassifier(nn.Module):
    """CLIP-based classifier with a custom classification head."""
    def __init__(self, clip_model, num_classes: int) -> None:
        super().__init__()
        self.clip = clip_model
        
        # Get embedding dimension from clip model
        if hasattr(clip_model.visual, "output_dim"):
            embedding_dim = clip_model.visual.output_dim
        elif hasattr(clip_model.visual, "trunk"):
            if hasattr(clip_model.visual.trunk, "embed_dim"):
                embedding_dim = clip_model.visual.trunk.embed_dim
            elif hasattr(clip_model.visual, "head"):
                embedding_dim = clip_model.visual.head.proj.out_features
            else:
                raise ValueError("Could not find embedding dimension in clip model")
        else:
            raise ValueError("Could not find embedding dimension in clip model")
        
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        image_features = self.clip.encode_image(images)
        return self.classifier(image_features)


class BatchCollator:
    def __init__(self, processor: Callable):
        self.processor = processor

    def __call__(self, batch):
        images = [dp["image"] for dp in batch]
        genders = [dp["gender"] for dp in batch]
        images = torch.stack([self.processor(image) for image in images])
        gender = torch.tensor(genders, dtype=torch.long)
        return images, gender


@torch.no_grad()
def evaluate(
    model: CLIPGenderClassifier,
    dataloader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Evaluate model on validation set and compute metrics."""
    model.eval()  # Set model to evaluation mode

    # Initialize metrics
    total_loss, n = 0, 0
    y_true, y_pred = [], []
    uncertainty = []
    
    # Iterate over validation set
    for images, labels in tqdm(dataloader, desc='Evaluating'):
        # Catch edge case where all samples are invalid
        if images is None or labels is None:
            continue
        
        # Get class logits and ground truth labels
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        y_true.extend(labels.flatten().cpu().tolist())
        
        # Compute loss and uncertainty for cross-entropy loss
        y_pred_batch = torch.argmax(logits, dim=1).flatten().cpu().tolist()
        loss = nn.CrossEntropyLoss()(logits, labels)
        total_loss += loss.item() * images.size(0)
        # Uncertainty is 1 - max softmax probability
        uncertainty_batch = (1 - torch.softmax(logits, dim=-1).max(dim=-1).values).flatten().cpu().tolist()
        uncertainty.extend(uncertainty_batch)
        
        y_pred.extend(y_pred_batch)
        n += images.size(0)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mispredictions = [y_true[i] != y_pred[i] for i in range(len(y_true))]
    auc_roc = roc_auc_score(mispredictions, uncertainty)
    mean_uncertainty = np.mean(uncertainty).item()
    
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy,
        "kappa": kappa,
        "auc_roc": auc_roc,
        "mean_uncertainty": mean_uncertainty,
    }


def train_one_epoch(
    model: CLIPGenderClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    global_step: int,
    scheduler: OneCycleLR = None,
) -> tuple[float, int]:
    """Train the model for one epoch."""
    model.train()  # Set model to training mode

    # Initialize metrics
    total_loss, n = 0, 0
    running_loss = None
    running_accuracy, running_kappa = None, None
    
    # Iterate over training set
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        # Catch edge case where all samples are invalid
        if images is None or labels is None:
            continue
        
        # Move images and labels to device
        images, labels = images.to(device), labels.to(device)

        # Get class logits
        logits = model(images)
        
        # Compute loss and uncertainty for cross-entropy loss
        loss = nn.CrossEntropyLoss()(logits, labels)
        y_pred_batch = torch.argmax(logits, dim=1).flatten().cpu().tolist()
        
        # Zero gradients, backpropagate, and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        y_true_batch = labels.flatten().cpu().tolist()
        batch_kappa = cohen_kappa_score(y_true_batch, y_pred_batch)
        batch_accuracy = accuracy_score(y_true_batch, y_pred_batch)
        running_accuracy = batch_accuracy if running_accuracy is None else 0.95 * running_accuracy + 0.05 * batch_accuracy
        running_kappa = batch_kappa if running_kappa is None else 0.95 * running_kappa + 0.05 * batch_kappa
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        global_step += 1
        
        # Update progress bar with moving average of metrics
        running_loss = loss.item() if running_loss is None else 0.95 * running_loss + 0.05 * loss.item()
        pbar.set_postfix({'kappa': 100 * running_kappa, 'accuracy': 100 * running_accuracy, 'lr': optimizer.param_groups[0]['lr']})
    
    avg_loss = total_loss / max(n, 1)
    return avg_loss, global_step


def load_model(args: argparse.Namespace) -> tuple[CLIPGenderClassifier, Callable]:
    """Load the OpenCLIP model and preprocessing function."""
    model_name, pretrained = args.model.split('/')
    clip_model, _, processor = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    return clip_model, processor


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> tuple[nn.Module, torch.optim.Optimizer, int, dict]:
    """Load model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(path, map_location='cpu')
        
    model.load_state_dict(checkpoint['model_state_dict'])
        
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    epoch = checkpoint.get('epoch', None)
    metrics = checkpoint.get('metrics', {})
        
    return model, optimizer, epoch, metrics


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
) -> None:
    """Save model and optimizer state."""
    checkpoint = {
        'epoch': epoch,
        'metrics': metrics,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint['model_state_dict'] = model.state_dict()
    torch.save(checkpoint, path)


def get_optimizer_and_scheduler(
    args: argparse.Namespace,
    params: list[nn.Parameter],
    steps_per_epoch: int,
    total_epochs: int,
) -> tuple[torch.optim.Optimizer, OneCycleLR]:
        """Helper to create optimizer and OneCycleLR scheduler."""
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        total_steps = total_epochs * steps_per_epoch
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=args.lr / args.min_lr,
        )
        return optimizer, scheduler


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ViT-B-32/laion400m_e32')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Initial learning rate for OneCycleLR scheduler.')
    parser.add_argument('--output', type=str, default='results_scratch/clip_finetuned_gender')
    parser.add_argument('--train-cls-layer-first', action='store_true', help='Train only the classifier layer in the first epoch, then all parameters.')
    args = parser.parse_args()

    # Prepare output directory for experiment results
    model_name = args.model.split('/')[0]
    experiment_name = f'{model_name}'
    experiment_name += f'batch{args.batch_size}_epochs{args.epochs}_lr{args.lr}'
    if args.train_cls_layer_first:
        experiment_name += '_clsfirst'
    experiment_directory = os.path.join(args.output, experiment_name)
    os.makedirs(experiment_directory, exist_ok=True)

    if os.path.exists(os.path.join(experiment_directory, 'log.csv')):
        logging.info(f"Model already trained. Exiting.")
        sys.exit(0)

    # Load model and datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, processor = load_model(args)
    hf_token = os.getenv("HFTOKEN")
    train_dataset = load_dataset("LGirrbach/gender-dataset-v1", split="train", token=hf_token)
    val_dataset = load_dataset("LGirrbach/gender-dataset-v1", split="val", token=hf_token)
    collate_fn = BatchCollator(processor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)
    clip_model.to(device)

    num_classes = len(train_dataset.features["gender"].names)
    classifier = CLIPGenderClassifier(clip_model, num_classes).to(device)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch

    # Set up optimizer/scheduler for first epoch
    if args.train_cls_layer_first:
        # Freeze backbone, train only classifier head in first epoch
        for p in classifier.clip.parameters():
            p.requires_grad = False
        for p in classifier.classifier.parameters():
            p.requires_grad = True
        # Use constant lr=1e-3 for classifier layer in first epoch, no scheduler
        optimizer = torch.optim.AdamW(classifier.classifier.parameters(), lr=1e-3)
        scheduler = None
    else:
        optimizer, scheduler = get_optimizer_and_scheduler(args, classifier.parameters(), steps_per_epoch, args.epochs)

    best_val_accuracy = 0.0
    global_step = 0
    os.makedirs(args.output, exist_ok=True)
    log = []

    for epoch in range(args.epochs):
        # If using train-cls-layer-first, unfreeze backbone and switch optimizer after first epoch
        if args.train_cls_layer_first and epoch == 1:
            for p in classifier.clip.parameters():
                p.requires_grad = True
            # Re-initialize optimizer and scheduler for all parameters, start scheduler now
            optimizer, scheduler = get_optimizer_and_scheduler(
                args,
                classifier.parameters(),
                steps_per_epoch,
                args.epochs - 1,
            )
        # Train one epoch
        avg_train_loss, global_step = train_one_epoch(
            classifier, train_loader, optimizer, device, global_step, scheduler
        )
        # Evaluate the model
        validation_metrics = evaluate(classifier, val_loader, device)
        log.append({'epoch': epoch+1, 'train_loss': avg_train_loss, **validation_metrics})
        # Log metrics for this epoch
        loss_msg = f'train_loss={avg_train_loss:.4f}'
        kappa_msg = f'kappa={validation_metrics["kappa"]:.4f}'
        accuracy_msg = f'accuracy={validation_metrics["accuracy"]:.4f}'
        auc_roc_msg = f'auc_roc={validation_metrics["auc_roc"]:.4f}'
        mean_uncertainty_msg = f'mean_uncertainty={validation_metrics["mean_uncertainty"]:.4f}'
        logging.info(f'Epoch {epoch+1}: {loss_msg}, {kappa_msg}, {accuracy_msg}, {auc_roc_msg}, {mean_uncertainty_msg}')
        
        # Save the best model checkpoint if accuracy improved
        if validation_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = validation_metrics['accuracy']
            save_checkpoint(
                os.path.join(experiment_directory, 'best.ckpt'),
                classifier,
                optimizer,
                epoch+1,
                validation_metrics
            )
    
    # Save final checkpoint after training
    save_checkpoint(
        os.path.join(experiment_directory, 'last.ckpt'),
        classifier,
        optimizer,
        args.epochs,
        log[-1] if log else {}
    )
    # Save training log to CSV
    pd.DataFrame(log).to_csv(os.path.join(experiment_directory, 'log.csv'), index=False)

if __name__ == '__main__':
    main()
