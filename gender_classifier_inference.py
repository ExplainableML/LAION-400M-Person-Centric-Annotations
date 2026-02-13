import os
import sys
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from train_gender_classifier import (
    load_model,
    CLIPGenderClassifier,
    BatchCollator,
)
from torch.utils.data import DataLoader


def construct_experiment_directory(args):
    model_name = args.model.split('/')[0]
    experiment_name = f'{model_name}'
    experiment_name += f'batch{args.batch_size}_epochs{args.epochs}_lr{args.lr}'
    if args.train_cls_layer_first:
        experiment_name += '_clsfirst'
    experiment_directory = os.path.join(args.output, experiment_name)
    return experiment_directory


def run_inference(dataloader, classifier, idx_to_gender, gender_names, device) -> pd.DataFrame:
    results = []
    for batch in tqdm(dataloader, desc='Inference'):
        # Unpack batch and skip if there are no images or labels
        images, labels = batch
        if images is None or labels is None:
            continue
        
        # Move images to device and get logits
        images = images.to(device)
        logits = classifier(images)
        
        # Get predictions and convert to numpy
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        labels = labels.cpu().tolist()
        # Convert predictions and labels to class names
        preds = [idx_to_gender[pred] for pred in preds]
        labels = [idx_to_gender[label] for label in labels]

        # Convert logits to float
        logits = logits.cpu().numpy().tolist()

        # Save results
        for predicted_gender, ground_truth_gender, logits in zip(preds, labels, logits):
            logits_dict = {f"logit_{gender_names[i]}": logit for i, logit in enumerate(logits)}
            row = {
                "Ground Truth Label": ground_truth_gender,
                "Predicted Label": predicted_gender,
                **logits_dict
            }
            results.append(row)

    return pd.DataFrame.from_records(results)


@torch.no_grad()
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ViT-B-32/laion400m_e32')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--output', type=str, default='results_scratch/clip_finetuned_gender')
    parser.add_argument('--train-cls-layer-first', action='store_true')
    args = parser.parse_args()
    
    # Make experiment directory
    experiment_directory = construct_experiment_directory(args)
    checkpoint_path = os.path.join(experiment_directory, 'best.ckpt')

    # Check if checkpoint exists, exit if not
    if not os.path.exists(checkpoint_path):
        print(f"No trained model found at {checkpoint_path}. Exiting.")
        sys.exit(0)
    
    # Initialize model and datasets
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, processor = load_model(args)

    # Load datasets
    hf_token = os.getenv("HFTOKEN")
    val_dataset = load_dataset("LGirrbach/gender-dataset-v1", split="val", token=hf_token)
    # Try to load a test-like split; fall back to train if no dedicated test split
    try:
        test_dataset = load_dataset("LGirrbach/gender-dataset-v1", split="test", token=hf_token)
    except Exception:
        test_dataset = load_dataset("LGirrbach/gender-dataset-v1", split="train", token=hf_token)

    collate_fn = BatchCollator(processor)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)

    # Build classifier
    num_classes = len(val_dataset.features["gender"].names)
    classifier = CLIPGenderClassifier(clip_model, num_classes).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    # Make mapping from gender index to gender name (string)
    gender_names = val_dataset.features["gender"].names
    idx_to_gender = {idx: name for idx, name in enumerate(gender_names)}

    # Run inference
    val_results = run_inference(val_loader, classifier, idx_to_gender, gender_names, device)
    test_results = run_inference(test_loader, classifier, idx_to_gender, gender_names, device)

    val_output_csv = os.path.join(experiment_directory, 'val_inference_results.csv')
    test_output_csv = os.path.join(experiment_directory, 'test_inference_results.csv')
    val_results.to_csv(val_output_csv, index=False)
    test_results.to_csv(test_output_csv, index=False)
    print(f"Inference results saved to {val_output_csv} and {test_output_csv}")

if __name__ == '__main__':
    main() 