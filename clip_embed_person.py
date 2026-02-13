import os
import sys
import torch
import datasets
import argparse
import open_clip
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from typing import Callable
from torch.utils.data import DataLoader, Dataset
from lexical_data.sobit_taxonomy import sobit_taxonomy
from benchmark.utils.phase_reader import PhaseImageReader
from open_clip.zero_shot_metadata import SIMPLE_IMAGENET_TEMPLATES
from open_clip.zero_shot_classifier import build_zero_shot_classifier


def pad_and_resize_with_bbox(
    image: Image.Image,
    bbox: list[float],
    target_size: int = 256,
    padding_color: tuple[int, int, int] = (0, 0, 0)
) -> tuple[Image.Image, list[float]]:
    """
    Pads and resizes an image to a target square size while maintaining 
    the original aspect ratio, and updates the corresponding bounding 
    box coordinates.

    Args:
        image (PIL.Image.Image): The input image in PIL format.
        bbox (list[float]): The input bounding box as a list [x1, y1, x2, y2].
        target_size (int): The side length for the output square image.
        padding_color (tuple[int, int, int]): The RGB color for the padding.

    Returns:
        tuple[PIL.Image.Image, list[float]]: A tuple containing:
            - The processed (padded and resized) image.
            - The updated bounding box coordinates [x1, y1, x2, y2].
    """
    # 1. Get original image dimensions
    original_width, original_height = image.size
    x1, y1, x2, y2 = bbox

    # 2. Calculate the scaling factor to fit the image into the target size
    #    while maintaining the aspect ratio.
    scale = target_size / max(original_width, original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 3. Resize the image using the calculated dimensions.
    #    Image.Resampling.LANCZOS is a high-quality downsampling filter.
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 4. Create a new image with the target size and padding color.
    #    This will serve as the background for the resized image.
    padded_image = Image.new(image.mode, (target_size, target_size), padding_color)

    # 5. Calculate the padding required to center the resized image.
    pad_x = (target_size - new_width) // 2
    pad_y = (target_size - new_height) // 2

    # 6. Paste the resized image onto the center of the padded background.
    padded_image.paste(resized_image, (pad_x, pad_y))

    # 7. Update the bounding box coordinates by applying the same scale
    #    and adding the padding offset.
    new_x1 = (x1 * scale) + pad_x
    new_y1 = (y1 * scale) + pad_y
    new_x2 = (x2 * scale) + pad_x
    new_y2 = (y2 * scale) + pad_y
    
    updated_bbox = [new_x1, new_y1, new_x2, new_y2]

    return padded_image, updated_bbox


class PhaseReader(PhaseImageReader):
    def __init__(self, path_to_data: str, processor: Callable = None, split: str = "all") -> None:
        super().__init__(path_to_data, processor=None, split=split)
        self.person_processor = processor
    
    def __len__(self):
        return self.total_samples

    def __iter__(self):
        for image, annotations in super().__iter__():
            for annotation in annotations:
                gender = annotation["person_gender"]
                race = annotation["person_ethnicity"]
                bbox = annotation["person_bbox"]

                image_resized, bbox_resized = pad_and_resize_with_bbox(image, bbox)
                person_crop = image_resized.crop(bbox_resized)

                if self.person_processor is not None:
                    person_crop = self.person_processor(person_crop)
                
                yield person_crop, {"race": race, "gender": gender}


class CausalFaceReader(Dataset):
    def __init__(self, path_to_data: str, processor: Callable = None) -> None:
        super().__init__()
        self.path_to_data = path_to_data
        self.processor = processor
        self.image_paths = []
        
        # Read paths to images
        for root, _, files in os.walk(path_to_data):
            for file in files:
                if "rm_bg" in file:
                    continue
                if file.endswith(".jpg") or file.endswith(".png"):
                    self.image_paths.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> tuple[Image.Image, dict]:
        image_path = self.image_paths[index]
        image_name = os.path.basename(image_path).split(".")[0]
        race, gender, *_ = image_name.split("_")
        seed = int(image_path.split("/")[-2].split("_")[-1])
        variation = image_path.split("/")[-3].split("_")[-1]
        
        image = Image.open(image_path).convert("RGB")
        if self.processor is not None:
            image = self.processor(image)
        
        return image, {"race": race, "gender": gender, "seed": seed, "variation": variation}

class GenderDataset(Dataset):
    def __init__(self, processor: Callable = None):
        super().__init__()
        self.processor = processor
        self.dataset = datasets.load_dataset("LGirrbach/gender-dataset-v1", split="train", token=os.environ["HFTOKEN"])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> tuple[Image.Image, dict]:
        dp = self.dataset[index]
        image = dp["image"]
        gender = self.dataset.features["gender"].int2str(dp["gender"])
        if self.processor is not None:
            image = self.processor(image)
        
        return image, {"gender": gender}

class RaceDataset(Dataset):
    def __init__(self, processor: Callable = None):
        super().__init__()
        self.processor = processor
        self.dataset = datasets.load_dataset("LGirrbach/race-dataset-v1", split="train", token=os.environ["HFTOKEN"])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> tuple[Image.Image, dict]:
        dp = self.dataset[index]
        image = dp["image"]
        race = self.dataset.features["race"].int2str(dp["race"])
        gender = self.dataset.features["gender"].int2str(dp["gender"])
        if self.processor is not None:
            image = self.processor(image)
        
        return image, {"race": race, "gender": gender}

        
def collate_fn(batch: list[tuple[Image.Image, dict]]) -> tuple[torch.Tensor, list[dict]]:
    images, annotations = zip(*batch)
    return torch.stack(images), annotations


if __name__ == "__main__":
    PHASE_PATH = "data/benchmarks/phase"
    CAUSALFACE_PATH = "data/benchmarks/causalface/images"

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["phase", "causalface", "gender", "race"])
    parser.add_argument("--model", type=str, required=True)  # "ViT-B-32/laion400m_e32"
    parser.add_argument("--categories", type=str, required=True, choices=["sobit", "guilbeault"])
    parser.add_argument("--output-dir", type=str, default="results/clip_scores/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA is required for this script"

    # Load social categories
    if args.categories == "sobit":
        categories = list(sorted(set.union(*[set(v) for v in sobit_taxonomy.values()])))
    elif args.categories == "guilbeault":
        categories = []
        with open("lexical_data/guilbeault_social_categories.txt", "r") as f:
            for line in f:
                categories.append(line.strip())
    else:
        raise ValueError(f"Invalid categories: {args.categories}")

    # Load the model
    model_name, pretrained = args.model.split("/")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model = model.eval()
    dtype = next(iter(model.parameters())).dtype
    tokenizer = open_clip.get_tokenizer(model_name)
    zeroshot_weights = build_zero_shot_classifier(
        model,
        tokenizer,
        classnames=categories,
        templates=SIMPLE_IMAGENET_TEMPLATES,
        device=device,
        use_tqdm=True
    )

    if args.text_only:
        text_embeddings = zeroshot_weights.cpu().numpy()
        save_path = os.path.join(args.output_dir, args.dataset, args.categories, args.model, "text_only")
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "text_embeddings.npy"), text_embeddings)
        sys.exit()

    
    # Create the dataset
    if args.dataset == "phase":
        dataset = PhaseReader(PHASE_PATH, processor=preprocess, split="train")
    elif args.dataset == "causalface":
        dataset = CausalFaceReader(CAUSALFACE_PATH, processor=preprocess)
    elif args.dataset == "gender":
        dataset = GenderDataset(processor=preprocess)
    elif args.dataset == "race":
        dataset = RaceDataset(processor=preprocess)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    # Get the embeddings
    all_image_embeddings, all_scores = [], []
    all_metadata = []

    with torch.no_grad():
        for images, annotations in tqdm(dataloader):
            image_embeddings = model.encode_image(images.to(device))
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            all_image_embeddings.append(image_embeddings.detach().cpu().numpy())
        
            scores = image_embeddings @ zeroshot_weights
            all_scores.append(scores.detach().cpu().numpy())
            all_metadata.extend(annotations)
    
    all_image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_metadata = pd.DataFrame.from_records(all_metadata)
    
    # Save the scores
    save_path = os.path.join(args.output_dir, args.dataset, args.categories, args.model)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "image_embeddings.npy"), all_image_embeddings)
    np.save(os.path.join(save_path, "scores.npy"), all_scores)
    np.save(os.path.join(save_path, "text_embeddings.npy"), zeroshot_weights.cpu().numpy())
    all_metadata.to_csv(os.path.join(save_path, "metadata.csv"), index=False)
