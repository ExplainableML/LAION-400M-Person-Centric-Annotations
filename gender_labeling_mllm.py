import os
import sys
import torch
import random
import argparse
import pandas as pd

from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from typing import Callable, Any
from torch.utils.data import DataLoader, Dataset
from lmdeploy.vl import load_image as lmdeploy_load_image
from lmdeploy import pipeline, TurbomindEngineConfig, VisionConfig, GenerationConfig


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GenderDataset(Dataset):
    def __init__(self,
        images_path: str,
        bounding_boxes_path: str,
        processor: Callable,
        start_index: int = 0,
        num_samples: int = None,
    ) -> None:
        self.images_path = images_path
        self.bounding_boxes_path = bounding_boxes_path
        self.processor = processor
        self.start_index = start_index
        self.num_samples = num_samples

        # Load bounding boxes
        self.bounding_boxes = []
        files = list(sorted(os.listdir(bounding_boxes_path)))
        files = files[self.start_index:]
        files = files[:self.num_samples]
        for file in files:
            if not file.endswith(".csv"):
                continue
            bounding_boxes = pd.read_csv(os.path.join(bounding_boxes_path, file))
            self.bounding_boxes.append(bounding_boxes)
        
        self.bounding_boxes = pd.concat(self.bounding_boxes).reset_index(drop=True)

    def __len__(self):
        return len(self.bounding_boxes)
    
    def __getitem__(self, index):
        row = self.bounding_boxes.iloc[index]
        gender = row["gender"]
        if gender == "gender is unclear / cannot tell / no person visible":
            gender = "unclear"
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        member_name = row["member_name"].replace(".jpg", "")
        tarball_name = row["tarball_name"]
        image_path = os.path.join(self.images_path, gender, f"{gender}_{member_name}_{x1}_{y1}_{x2}_{y2}.png")
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None
        
        image = self.processor(image)

        metadata = {
            "tarball_name": tarball_name,
            "member_name": member_name,
            "gender": gender,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }

        return image, metadata

class MLLMGenderLabeler:
    def __init__(self, model: str, device: str = "cuda") -> None:
        self.model = model
        self.device = device

        backend_config = TurbomindEngineConfig(session_len=8192)
        vision_config = VisionConfig(max_batch_size=16)
        self.generation_config = GenerationConfig(max_new_tokens=5, do_sample=False)
        self.pipe = pipeline(
            self.model,
            backend_config=backend_config,
            vision_config=vision_config,
        )

    @staticmethod
    def get_preprocessor() -> Callable:
        return lmdeploy_load_image
    
    @torch.no_grad()
    def __call__(self, images: list[Image.Image], metas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompts, letter2options = zip(*[make_prompt() for _ in images])
        prompts = [(prompt, image) for prompt, image in zip(prompts, images)]
        answers = self.pipe(prompts, gen_config=self.generation_config)
        answers = [answer.text.upper().strip().split()[0].strip() for answer in answers]
        
        responses = []
        for answer, letter2option, meta in zip(answers, letter2options, metas):
            meta = deepcopy(meta)
            gender = letter2option.get(answer, None)
            meta["predicted_gender"] = gender
            responses.append(meta)
        
        return responses
    

def make_prompt() -> tuple[str, dict]:
    question = "What is the gender of the person in the image? Only answer if you are sure and there are visual cues in the image about the person's gender. If you are not sure, choose the option 'gender is unclear / cannot tell / no person visible'."
    options = [
        "only male person visible",
        "only female person visible",
        "both male and female people visible",
        "gender is unclear / cannot tell / no person visible"
    ]
    instruction = "Answer with a single letter."

    random.shuffle(options)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter2option = {alphabet[i]: option for i, option in enumerate(options)}
    option_list = "\n".join([f"{alphabet[i]}. {option}" for i, option in enumerate(options)])
    prompt = f"{question}\n{option_list}\n\n{instruction}"
    return prompt, letter2option


def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    images, metas = zip(*batch)
    return list(images), list(metas)


if __name__ == "__main__":
    # Set random seed
    random.seed(42)

    # Assert that we have a GPU
    assert torch.cuda.is_available(), "CUDA is not available"
    device = "cuda"

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-path', type=str, default="data/gender_dataset/images/")
    parser.add_argument("--model", type=str, required=True, choices=["InternVL3-2B", "InternVL3-8B", "InternVL3-14B", "llava-1.6", "phi-3.5", "qwen-vl-3B", "qwen-vl-7B"])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output-path', type=str, default="results/gender_dataset/labeling/")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    # Make save path
    result_file_name = f"results_{args.start_index}_{args.num_samples}.csv"
    save_path = os.path.join(args.output_path, args.model, result_file_name)
    
    if os.path.exists(save_path):
        print(f"Results already exist at {save_path}, exiting")
        sys.exit()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load the model
    model_name = {
        "InternVL3-2B": "OpenGVLab/InternVL3-2B",
        "InternVL3-8B": "OpenGVLab/InternVL3-8B",
        "InternVL3-14B": "OpenGVLab/InternVL3-14B",
        "llava-1.6": "liuhaotian/llava-v1.6-vicuna-7b",
        "phi-3.5": "microsoft/Phi-3.5-vision-instruct",
        "qwen-vl-3B": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen-vl-7B": "Qwen/Qwen2.5-VL-7B-Instruct"
    }[args.model]
    
    model = MLLMGenderLabeler(model_name, device)
    processor = model.get_preprocessor()

    # Load the dataset
    dataset = GenderDataset(
        args.images_path,
        "data/gender_dataset/shards",
        processor,
        args.start_index,
        args.num_samples
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=collate_fn)

    # Run the model
    results = []
    total_samples = len(dataset.bounding_boxes)
    for images, metas in tqdm(dataloader, total=total_samples // args.batch_size + 1, desc="Labeling images"):
        responses = model(images, metas)
        results.extend(responses)
    
    if not results:
        print("No results found, exiting")
        sys.exit()
    
    # Save the results
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")
