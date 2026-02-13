import os
import sys
import json
import torch
import argparse
import open_clip
import pandas as pd

from tqdm import tqdm
from typing import Callable
from torch.utils.data import DataLoader
from train_race_classifier import CLIPRaceClassifier
from utils.laion_reader import Laion400mBoundingBoxDataset


open_clip_model_mapping = {
    "ViT-B-16-SigLIP": "ViT-B-16-SigLIP/webli",
}

race_to_idx = {
    'unclear': 0,
    'southasian': 1,
    'disagreement': 2,
    'southeastasian': 3,
    'black': 4,
    'eastasian': 5,
    'white': 6,
    'latino': 7,
    'middleeastern': 8,
}

idx_to_race = {v: k for k, v in race_to_idx.items()}
race_names = list(race_to_idx.keys())


def load_model(model: str) -> tuple[CLIPRaceClassifier, Callable]:
    """Load the OpenCLIP model and preprocessing function."""
    model_name = model.split("_")[0]
    model = open_clip_model_mapping.get(model_name, None)
    if model is None:
        raise ValueError(f"Model {model_name} not found in open_clip_model_mapping")
    
    model_name, pretrained = model.split('/')
    clip_model, _, processor = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    return clip_model, processor


def collate_fn(batch):
    images, metas = zip(*batch)
    return torch.stack(list(images)), list(metas)


if __name__ == "__main__":
    # Assert that we have a GPU
    assert torch.cuda.is_available(), "CUDA is not available"
    device = "cuda"

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="ViT-B-16-SigLIP_16_20_1e-05")
    parser.add_argument('--path', type=str, default="/lustre/groups/eml/datasets/laion400m/laion400m-data")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--output-path', type=str, default="results/race_labeling/classifier/laion")
    parser .add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Make output directory
    result_path = os.path.join(args.output_path, args.model)
    result_filename = f"results_{args.start_idx}_{args.num_samples}.csv"
    os.makedirs(result_path, exist_ok=True)
    if not args.overwrite and os.path.exists(os.path.join(result_path, result_filename)):
        print("Result file already exists. Exiting.")
        sys.exit()

    # Load the model
    checkpoint_path = os.path.join("data/assets/race_classifier/checkpoints", f"{args.model}.ckpt")
    clip_model, processor = load_model(args.model)
    model = CLIPRaceClassifier(clip_model, 9).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load the dataset
    dataset = Laion400mBoundingBoxDataset(
        laion_400m_path=args.path,
        bounding_boxes_path="data/person_detection/bounding_boxes_filtered",
        processor=processor,
        start_idx=args.start_idx,
        num_samples=args.num_samples
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # Load thresholds
    with open("data/assets/race_classifier/spec_to_threshold_forsave.json", "r") as f:
        thresholds = json.load(f)
    
    threshold = thresholds[args.model]

    # Run the model
    results = []
    for images, metas in tqdm(dataloader):
        # Move images to device and get logits
        images = images.to(device)

        with torch.no_grad():
            logits = model(images)
        
        # Get predictions and convert to numpy
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        # Convert predictions to class names
        preds = [idx_to_race[pred] for pred in preds]

        # Convert logits to float
        logits = logits.cpu().numpy().tolist()

        # Save results
        for meta, predicted_race, logits in zip(metas, preds, logits):
            if max(logits) <= threshold:
                predicted_race = "unclear"

            logits_dict = {f"logit_{race_names[i]}": float(logit) for i, logit in enumerate(logits)}
            results.append(
                {
                    **meta,
                    "predicted_race": predicted_race,
                    **logits_dict,
                }
            )

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(result_path, result_filename), index=False)
