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
from train_gender_classifier import CLIPGenderClassifier
from utils.laion_reader import Laion400mBoundingBoxDataset


open_clip_model_mapping = {
    "ViT-B-16-SigLIP": "ViT-B-16-SigLIP/webli",
}

gender_to_idx = {
    'male': 0,
    'female': 1,
    'unclear': 2,
    'mixed': 3
}

idx_to_gender = {v: k for k, v in gender_to_idx.items()}
# Names will be adjusted dynamically if checkpoint indicates a different number of classes

def load_model(model: str) -> tuple[CLIPGenderClassifier, Callable]:
    """Load the OpenCLIP model and preprocessing function."""
    model_name = model.split("_")[0]
    mapped = open_clip_model_mapping.get(model_name, None)
    if mapped is None:
        raise ValueError(f"Model {model_name} not found in open_clip_model_mapping")
    model_name, pretrained = mapped.split('/')
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
    parser.add_argument('--model', type=str, default="ViT-B-16-SigLIP_32_2_5e-06")
    parser.add_argument('--path', type=str, default="/lustre/groups/eml/datasets/laion400m/laion400m-data")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--output-path', type=str, default="results/gender_labeling/classifier/laion")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Make output directory
    # Note: keep the same filename pattern as race script for consistency
    result_path = os.path.join(args.output_path, args.model)
    result_filename = f"results_{args.start_idx}_{args.num_samples}.csv"
    os.makedirs(result_path, exist_ok=True)
    if not args.overwrite and os.path.exists(os.path.join(result_path, result_filename)):
        print("Result file already exists. Exiting.")
        sys.exit()

    # Load the model backbone and processor
    clip_model, processor = load_model(args.model)

    # Resolve checkpoint path and inspect for class count
    checkpoint_dir = os.path.join("data/assets/gender_classifier/checkpoints")
    checkpoint_filename = args.model if args.model.endswith('.ckpt') else f"{args.model}.ckpt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    checkpoint_cpu = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Infer number of classes from checkpoint
    if 'model_state_dict' not in checkpoint_cpu:
        raise ValueError("Checkpoint missing 'model_state_dict'")
    state_dict = checkpoint_cpu['model_state_dict']
    if 'classifier.weight' not in state_dict:
        raise ValueError("Checkpoint missing 'classifier.weight' in state_dict")
    num_classes = state_dict['classifier.weight'].shape[0]

    # Assert that the number of classes in the checkpoint matches our label mapping
    assert num_classes == len(gender_to_idx), \
        f"Number of classes in checkpoint ({num_classes}) does not match gender_to_idx ({len(gender_to_idx)})"

    # Build classifier and load weights
    model = CLIPGenderClassifier(clip_model, num_classes).to(device)
    model.load_state_dict(state_dict)
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
        preds = [idx_to_gender.get(pred, str(pred)) for pred in preds]

        # Convert logits to float
        logits = logits.cpu().numpy().tolist()

        # Save results
        for meta, predicted_gender, sample_logits in zip(metas, preds, logits):
            # Build a dict of logits with stable column names
            logits_dict = {f"logit_{idx_to_gender.get(i, str(i))}": float(logit) for i, logit in enumerate(sample_logits)}
            results.append(
                {
                    **meta,
                    "predicted_gender": predicted_gender,
                    **logits_dict,
                }
            )

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(result_path, result_filename), index=False) 