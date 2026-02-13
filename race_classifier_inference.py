import os
import sys
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from train_race_classifier import (
    load_model,
    RaceClassifier,
    CLIPRaceClassifier,
)
from torch.utils.data import DataLoader


def construct_experiment_directory(args):
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
    return experiment_directory


def run_inference(dataloader, classifier, idx_to_race, race_names, device) -> pd.DataFrame:
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
        labels = torch.argmax(labels, dim=1).cpu().tolist()
        # Convert predictions and labels to class names
        preds = [idx_to_race[pred] for pred in preds]
        labels = [idx_to_race[label] for label in labels]

        # Convert logits to float
        logits = logits.cpu().numpy().tolist()

        # Save results
        for predicted_race, ground_truth_race, logits in zip(preds, labels, logits):
            logits_dict = {f"logit_{race_names[i]}": logit for i, logit in enumerate(logits)}
            row = {
                "Ground Truth Label": ground_truth_race,
                "Predicted Label": predicted_race,
                **logits_dict
            }
            results.append(row)

    return pd.DataFrame.from_records(results)


@torch.no_grad()
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ViT-B-32/laion400m_e32')
    parser.add_argument('--dataset', type=str, default='raceclassifier')
    parser.add_argument('--loss-type', type=str, default='evidential', choices=['evidential', 'cross-entropy'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--annealing-ratio', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='results_scratch/clip_finetuned_race')
    parser.add_argument('--train-cls-layer-first', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--filter-disagreement', action='store_true')
    parser.add_argument('--debug', action='store_true')
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
    val_dataset = RaceClassifier(
        split='val',
        preprocess_image=processor,
        filter_disagreement=False,
        debug=False
    )
    test_dataset = RaceClassifier(
        split='calibration',
        preprocess_image=processor,
        filter_disagreement=False,
        debug=False
    )       
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    classifier = CLIPRaceClassifier(clip_model, val_dataset.num_classes).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    # Make mapping from race index to race name (string)
    idx_to_race = {idx: race for race, idx in val_dataset.race_to_idx.items()}
    race_names = val_dataset.race_classes

    # Run inference
    val_results = run_inference(val_loader, classifier, idx_to_race, race_names, device)
    test_results = run_inference(test_loader, classifier, idx_to_race, race_names, device)

    val_output_csv = os.path.join(experiment_directory, 'val_inference_results.csv')
    test_output_csv = os.path.join(experiment_directory, 'test_inference_results.csv')
    val_results.to_csv(val_output_csv, index=False)
    test_results.to_csv(test_output_csv, index=False)
    print(f"Inference results saved to {val_output_csv} and {test_output_csv}")

if __name__ == '__main__':
    main()