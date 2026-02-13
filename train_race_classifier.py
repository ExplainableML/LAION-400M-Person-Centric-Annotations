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
from PIL import Image
from torch import Tensor
from typing import Callable
from datasets import load_dataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
from utils.evidential_classification.loss import mse_loss, softplus_evidence

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging for progress and results
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class PhaseDataset(Dataset):
    """Custom dataset for the PHASE benchmark, with balanced sampling for training."""
    def __init__(self, path_to_data: str = "data/benchmarks/phase/", split: str = "train", preprocess_image: Callable = None) -> None:
        # Store arguments
        self.path_to_data = path_to_data
        self.split = split
        self.preprocess_image = preprocess_image

        # Make sure preprocess image function is provided
        assert self.preprocess_image is not None, "Preprocess image function must be provided"

        # Load and filter annotations
        self.dataset = pd.read_csv(os.path.join(self.path_to_data, "annotations", "phase_annotations.csv"))
        self.dataset = self.dataset[~self.dataset["ethnicity"].isin(["other", "disagreement", "unsure"])]
        self.dataset = self.dataset[self.dataset["split"] == self.split]
        self.race_classes = list(sorted(self.dataset["ethnicity"].unique().tolist()))
        self.num_classes = len(self.race_classes)
        self.race_to_idx = {race: idx for idx, race in enumerate(self.race_classes)}
        
        # Balance classes for training
        if self.split == "train":
            race_datasets = {race: df for race, df in self.dataset.groupby("ethnicity")}
            max_subset_size = max(len(df) for df in race_datasets.values())
            race_datasets = {race: df.sample(max_subset_size // 4, replace=len(df) < max_subset_size // 4) for race, df in race_datasets.items()}
            self.dataset = pd.concat(race_datasets.values())

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # Load image and label for a given index
        item = self.dataset.iloc[idx]
        image_id, split = item["image_id"], item["split"]
        x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
        race = item["ethnicity"]
        image_path = os.path.join(self.path_to_data, "images", split, str(image_id))
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            return None, None
        image = image.crop((x1, y1, x2, y2))
        image = self.preprocess_image(image)
        race_idx = self.race_to_idx[race]
        race_idx = torch.tensor(race_idx).long()
        race_one_hot = torch.nn.functional.one_hot(race_idx, num_classes=self.num_classes)
        return image, race_one_hot


class RaceClassifier(Dataset):
    """Dataset for our own Race Stereotype dataset."""
    def __init__(
        self,
        split: str = "train",
        preprocess_image: Callable = None,
        filter_disagreement: bool = False,
        debug: bool = False,
    ) -> None:
        self.split = split
        self.preprocess_image = preprocess_image
        self.filter_disagreement = filter_disagreement
        self.debug = debug
        assert self.preprocess_image is not None, "Preprocess image function must be provided"
        
        # Load dataset from HuggingFace with token
        hf_token = os.environ.get("HFTOKEN", None)
        if hf_token is None:
            raise ValueError("HFTOKEN environment variable must be set to access the dataset.")
        
        self.dataset = load_dataset("LGirrbach/race-dataset-v1", split=split, token=hf_token)
        if self.debug:
            self.dataset = self.dataset.select(range(1000))
        
        # Filter out disagreement for compatibility with PHASE dataset
        if self.filter_disagreement:
            self.dataset = self.dataset.filter(lambda x: x["race"] != "disagreement")
        
        # Get race classes and make mapping from race to index (mainly for compatibility with PHASE dataset)
        self.race_classes = list(sorted(list(set(self.dataset["race"]))))
        self.race_classes = [self.dataset.features["race"].int2str(race) for race in self.race_classes]
        self.num_classes = len(self.race_classes)
        self.race_to_idx = {race: idx for idx, race in enumerate(self.race_classes)}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # Retrieve datapoint
        item = self.dataset[idx]
        
        # Preprocess image
        image = item["image"].convert("RGB")
        image = self.preprocess_image(image)
        
        # Get race index and convert to one-hot encoding
        race = item["race"]
        race = self.dataset.features["race"].int2str(race)
        race = "unclear" if race == "disagreement" else race
        race_idx = self.race_to_idx[race]
        race_idx = torch.tensor(race_idx).long()
        race_one_hot = torch.nn.functional.one_hot(race_idx, num_classes=self.num_classes)

        return image, race_one_hot


class CLIPRaceClassifier(nn.Module):
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


def collate_fn(batch):
    """Collate function to filter out None samples and stack tensors."""
    image_tensors, race_one_hots = zip(*batch)
    valid_images, valid_labels = [], []
    
    # Filter out invalid samples -> Allows dataloading to fail on invalid images
    for image_tensor, race_one_hot in zip(image_tensors, race_one_hots):
        if image_tensor is None or race_one_hot is None:
            continue
        valid_images.append(image_tensor)
        valid_labels.append(race_one_hot)
    
    # Catch edge case where all samples are invalid
    if not valid_images:
        return None, None
    
    images = torch.stack(valid_images)
    labels = torch.stack(valid_labels).float()
    
    return images, labels


@torch.no_grad()
def evaluate(
    model: CLIPRaceClassifier,
    dataloader: DataLoader,
    device: str,
    global_step: int,
    annealing_step: int,
    loss_type: str,
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
        y_true_batch = torch.argmax(labels, dim=1).flatten().cpu().tolist()
        y_true.extend(y_true_batch)
        
        # Compute loss and uncertainty for evidential loss
        if loss_type == "evidential":
            # Calculate class-wise modes from dirichlet distribution
            evidence = softplus_evidence(logits)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            p_hat = torch.div(alpha, S)
            
            # Get predictions
            y_pred_batch = torch.argmax(p_hat, dim=1).flatten().cpu().tolist()
            
            # Compute loss
            loss = mse_loss(labels, alpha, global_step, annealing_step)
            total_loss += loss.mean().item() * images.size(0)
            
            # Calculate uncertainty
            K = logits.shape[-1]
            uncertainty_batch = torch.div(K, S).flatten().cpu().tolist()
            uncertainty.extend(uncertainty_batch)
        
        # Compute loss and uncertainty for cross-entropy loss
        elif loss_type == "cross-entropy":
            y_pred_batch = torch.argmax(logits, dim=1).flatten().cpu().tolist()
            loss = nn.CrossEntropyLoss()(logits, torch.argmax(labels, dim=1))
            total_loss += loss.item() * images.size(0)
            # Uncertainty is 1 - max softmax probability
            uncertainty_batch = (1 - torch.softmax(logits, dim=-1).max(dim=-1).values).flatten().cpu().tolist()
            uncertainty.extend(uncertainty_batch)
        else:
            # Catch unknown loss type
            raise ValueError(f"Unknown loss type: {loss_type}")
        
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
    model: CLIPRaceClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    global_step: int,
    annealing_step: int,
    scheduler: OneCycleLR = None,
    loss_type: str = "evidential",
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

        # Compute loss and uncertainty for evidential loss
        if loss_type == "evidential":
            evidence = softplus_evidence(logits)
            alpha = evidence + 1
            loss = mse_loss(labels, alpha, global_step, annealing_step)
            loss = loss.mean()
            S = torch.sum(alpha, dim=1, keepdim=True)
            p_hat = torch.div(alpha, S)
            y_pred_batch = torch.argmax(p_hat, dim=1).flatten().cpu().tolist()
        
        # Compute loss and uncertainty for cross-entropy loss
        elif loss_type == "cross-entropy":
            loss = nn.CrossEntropyLoss()(logits, torch.argmax(labels, dim=1))
            y_pred_batch = torch.argmax(logits, dim=1).flatten().cpu().tolist()
        
        else:
            # Catch unknown loss type
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Zero gradients, backpropagate, and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        y_true_batch = torch.argmax(labels, dim=1).flatten().cpu().tolist()
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


def load_model(args: argparse.Namespace) -> tuple[CLIPRaceClassifier, Callable]:
    """Load the OpenCLIP model and preprocessing function."""
    model_name, pretrained = args.model.split('/')
    clip_model, _, processor = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    return clip_model, processor

def load_datasets(
    args: argparse.Namespace,
    processor: Callable,
    filter_disagreement: bool = False,
    debug: bool = False,
) -> tuple[Dataset, Dataset]:
    """Load the appropriate dataset based on args.dataset."""
    # Load phase dataset
    if args.dataset == 'phase':
        train_dataset = PhaseDataset(split='train', preprocess_image=processor)
        val_dataset = PhaseDataset(split='val', preprocess_image=processor)

        # Synchronize number of classes between train and val sets
        max_classes_split, num_classes = max(
            ("train", train_dataset.num_classes),
            ("val", val_dataset.num_classes),
            key=lambda x: x[1]
        )
        train_dataset.num_classes = num_classes
        val_dataset.num_classes = num_classes
        
        # Re-assign race classes and indices to ensure they match
        if max_classes_split == "train":
            val_dataset.race_classes = train_dataset.race_classes
            val_dataset.race_to_idx = train_dataset.race_to_idx
        else:
            train_dataset.race_classes = val_dataset.race_classes
            train_dataset.race_to_idx = val_dataset.race_to_idx
        
        return train_dataset, val_dataset
    
    # Load race classifier dataset
    elif args.dataset == 'raceclassifier':
        train_dataset = RaceClassifier(
            split='train',
            preprocess_image=processor,
            filter_disagreement=filter_disagreement,
            debug=debug
        )
        val_dataset = RaceClassifier(
            split='val',
            preprocess_image=processor,
            filter_disagreement=filter_disagreement,
            debug=debug
        )

        # Synchronize number of classes between train and val sets
        max_classes_split, num_classes = max(
            ("train", train_dataset.num_classes),
            ("val", val_dataset.num_classes),
            key=lambda x: x[1]
        )
        train_dataset.num_classes = num_classes
        val_dataset.num_classes = num_classes
        
        # Re-assign race classes and indices to ensure they match
        if max_classes_split == "train":
            val_dataset.race_classes = train_dataset.race_classes
            val_dataset.race_to_idx = train_dataset.race_to_idx
        else:
            train_dataset.race_classes = val_dataset.race_classes
            train_dataset.race_to_idx = val_dataset.race_to_idx
        return train_dataset, val_dataset
    
    else:
        # Catch unknown dataset
        raise ValueError(f"Dataset {args.dataset} not supported")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    lora_applied: bool = False,
) -> tuple[nn.Module, torch.optim.Optimizer, int, dict]:
        """Load model and optimizer state from a checkpoint file."""
        checkpoint = torch.load(path, map_location='cpu')
        
        if lora_applied and get_peft_model_state_dict is not None:
            # For LoRA, use the PEFT state dict loader
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
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
    lora_applied: bool,
) -> None:
        """Save model and optimizer state, handling LoRA if needed."""
        checkpoint = {
            'epoch': epoch,
            'metrics': metrics,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if lora_applied and get_peft_model_state_dict is not None:
            checkpoint['model_state_dict'] = get_peft_model_state_dict(model)
        else:
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
    parser.add_argument('--dataset', type=str, default='phase', choices=['phase','raceclassifier'])
    parser.add_argument('--loss-type', type=str, default='evidential', choices=['evidential', 'cross-entropy'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Initial learning rate for OneCycleLR scheduler (will be scaled by div_factor).')
    parser.add_argument('--annealing-ratio', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='results_scratch/clip_finetuned_race')
    parser.add_argument('--train-cls-layer-first', action='store_true', help='Train only the classifier layer in the first epoch, then all parameters.')
    parser.add_argument('--lora', action='store_true', help='If set, use LoRA adapters for ViT models (not for ResNet).')
    parser.add_argument('--filter-disagreement', action='store_true', help='If set, filter out disagreement from race dataset.')
    parser.add_argument("--debug", action='store_true', help='If set, run in debug mode.')
    args = parser.parse_args()

    # Prepare output directory for experiment results
    model_name = args.model.split('/')[0]
    experiment_name = f'{model_name}/{args.dataset}_{args.loss_type}/'
    experiment_name += f'batch{args.batch_size}_epochs{args.epochs}_lr{args.lr}'
    if args.loss_type == 'evidential':
        experiment_name += f'_annealing{args.annealing_ratio}'
    if args.train_cls_layer_first:
        experiment_name += '_clsfirst'
    if args.lora:
        experiment_name += '_lora'
    if args.filter_disagreement:
        experiment_name += '_filterdis'
    if args.debug:
        experiment_name += '_debug'
    experiment_directory = os.path.join(args.output, experiment_name)
    os.makedirs(experiment_directory, exist_ok=True)

    if os.path.exists(os.path.join(experiment_directory, 'log.csv')):
        logging.info(f"Model already trained. Exiting.")
        sys.exit(0)

    # Load model and datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, processor = load_model(args)
    train_dataset, val_dataset = load_datasets(args, processor, filter_disagreement=args.filter_disagreement, debug=args.debug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)
    clip_model.to(device)
    classifier = CLIPRaceClassifier(clip_model, train_dataset.num_classes).to(device)

    # LoRA logic: apply adapters to ViT backbones if requested
    lora_applied = False
    lora_params = None
    if args.lora:
        if LoraConfig is None:
            raise ImportError("peft library is required for LoRA support. Please install peft.")
        visual = getattr(clip_model, 'visual', None)
        is_vit = visual is not None and 'visiontransformer' in visual.__class__.__name__.lower()
        is_resnet = visual is not None and 'resnet' in visual.__class__.__name__.lower()
        # Find all linear layer names except classifier head
        linear_layer_names = [name.split('.')[-1] for name, module in classifier.named_modules() if isinstance(module, nn.Linear)]
        linear_layer_names = list(set(linear_layer_names))
        if "classifier" in linear_layer_names:
            linear_layer_names.remove("classifier")
        if is_vit:
            # Apply LoRA to all linear layers except classifier head
            lora_config = LoraConfig(
                r=8, lora_alpha=16, target_modules=linear_layer_names, lora_dropout=0.1, bias='none', task_type='FEATURE_EXTRACTION'
            )
            clip_model = get_peft_model(clip_model, lora_config)
            clip_model.print_trainable_parameters()
            lora_applied = True
            # Collect LoRA parameters for optimizer
            lora_params = list(clip_model.parameters())
            # Add classifier head params if not frozen
            lora_params += [p for n, p in classifier.named_parameters() if 'classifier' in n and p.requires_grad]
            print("[INFO] LoRA adapters applied to ViT visual backbone.")
        elif is_resnet:
            warnings.warn("LoRA adapters are not supported for ResNet backbones. Proceeding without LoRA.")
        else:
            warnings.warn("Unknown visual backbone. Proceeding without LoRA.")

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
        if lora_applied:
            optimizer, scheduler = get_optimizer_and_scheduler(args, lora_params, steps_per_epoch, args.epochs)
        else:
            optimizer, scheduler = get_optimizer_and_scheduler(args, classifier.parameters(), steps_per_epoch, args.epochs)

    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    global_step = 0
    total_training_steps = args.epochs * len(train_loader)
    annealing_step = int(total_training_steps * args.annealing_ratio)
    os.makedirs(args.output, exist_ok=True)
    log = []

    for epoch in range(args.epochs):
        # If using train-cls-layer-first, unfreeze backbone and switch optimizer after first epoch
        if args.train_cls_layer_first and epoch == 1:
            for p in classifier.clip.parameters():
                p.requires_grad = True
            # Re-initialize optimizer and scheduler for all parameters, start scheduler now
            if lora_applied:
                optimizer, scheduler = get_optimizer_and_scheduler(
                    args,
                    lora_params,
                    steps_per_epoch,
                    args.epochs - 1,
                )
            else:
                optimizer, scheduler = get_optimizer_and_scheduler(
                    args,
                    classifier.parameters(),
                    steps_per_epoch,
                    args.epochs - 1,
                )
        # Train one epoch
        avg_train_loss, global_step = train_one_epoch(
            classifier, train_loader, optimizer, device, global_step, annealing_step, scheduler, loss_type=args.loss_type
        )
        # Evaluate the model
        validation_metrics = evaluate(classifier, val_loader, device, global_step, annealing_step, loss_type=args.loss_type)
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
                validation_metrics,
                lora_applied
            )
    
    # Save final checkpoint after training
    save_checkpoint(
        os.path.join(experiment_directory, 'last.ckpt'),
        classifier,
        optimizer,
        args.epochs,
        log[-1] if log else {},
        lora_applied
    )
    # Save training log to CSV
    pd.DataFrame(log).to_csv(os.path.join(experiment_directory, 'log.csv'), index=False)

if __name__ == '__main__':
    main()
