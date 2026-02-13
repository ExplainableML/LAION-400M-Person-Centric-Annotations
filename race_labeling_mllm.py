import os
import sys
import torch
import random
import tarfile
import argparse
import pandas as pd

from tqdm import tqdm
from PIL import Image
from io import BytesIO
from copy import deepcopy
from typing import Callable, Any
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from lmdeploy.vl import load_image as lmdeploy_load_image
from lmdeploy import pipeline, TurbomindEngineConfig, VisionConfig, GenerationConfig


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RaceDataset(IterableDataset):
    def __init__(self, laion_path: str, bounding_boxes_path: str, processor: Callable, race: str = None, start_index: int = None, num_samples: int = None) -> None:
        self.laion_path = laion_path
        self.bounding_boxes_path = bounding_boxes_path
        self.processor = processor
        self.race = race
        self.start_index = start_index
        self.num_samples = num_samples

        # Load bounding boxes
        self.bounding_boxes = pd.read_csv(bounding_boxes_path)
        if self.race is not None:
            self.bounding_boxes = self.bounding_boxes[self.bounding_boxes["race"] == self.race]
        self.bounding_boxes = self.bounding_boxes.sort_values(by="image_id")

        if self.start_index is not None:
            self.bounding_boxes = self.bounding_boxes.iloc[self.start_index:]
        if self.num_samples is not None:
            self.bounding_boxes = self.bounding_boxes.iloc[:self.num_samples]

    def __len__(self):
        return len(self.bounding_boxes)
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            bounding_boxes = self.bounding_boxes.iloc[worker_info.id::worker_info.num_workers]
        else:
            bounding_boxes = self.bounding_boxes
        
        for tarball_id, tarball_id_df in bounding_boxes.groupby("tarball_id"):
            tarball_id = str(tarball_id).zfill(5) + ".tar"
            with tarfile.open(os.path.join(self.laion_path, tarball_id), "r") as tar:
                for _, row in tarball_id_df.iterrows():
                    # Load image
                    image_id = row["image_id"]
                    image_id_str = str(image_id).zfill(9) + ".jpg"
                    image = tar.extractfile(image_id_str)
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')

                    # Load bounding box and metadata
                    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
                    race = row["race"]
                    keyword = row["keyword"]

                    # Crop image
                    image_cropped = image.crop((x1, y1, x2, y2))

                    # Preprocess image
                    if self.processor is not None:
                        image_cropped = self.processor(image_cropped)

                    metadata = {
                        "tarball_id": tarball_id,
                        "image_id": image_id_str,
                        "keyword": keyword,
                        "race": race,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }

                    yield image_cropped, metadata


class HFRaceDataset(IterableDataset):
    def __init__(self, processor: Callable):
        self.processor = processor
        self.dataset = load_dataset("LGirrbach/race-dataset-v1", split="train", token=os.getenv("HFTOKEN"), streaming=True)
        
    def __iter__(self):
        for elem in self.dataset:
            image = elem["image"]
            if self.processor is not None:
                image = self.processor(image)
            metadata = {key: value for key, value in elem.items() if key != "image"}
            yield image, metadata
        
    def __len__(self):
        return len(self.dataset)


def normalize_race(race: str) -> str:
    race = race.lower().split(",")[0].strip()

    if race == "white":
        return "White / Caucasian"
    elif race == "black":
        return "Black / African American"
    elif race == "east asian":
        return "East Asian"
    elif race == "latino":
        return "Latino / Hispanic"
    elif race == "middle eastern":
        return "Middle Eastern / North African"
    elif race == "southeast asian":
        return "Southeast Asian"
    elif race == "indian":
        return "South Asian"
    elif race == "race is unclear / cannot tell / no person visible":
        return "race is unclear / cannot tell / no person visible"
    else:
        raise ValueError(f"Unknown race: {race}")

class MLLMRaceLabeler:
    def __init__(self, model: str, device: str = "cuda", include_unknown: bool = False) -> None:
        self.model = model
        self.device = device
        self.include_unknown = include_unknown

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
        prompts, letter2options = zip(*[make_prompt(self.include_unknown) for _ in images])
        prompts = [(prompt, image) for prompt, image in zip(prompts, images)]
        answers = self.pipe(prompts, gen_config=self.generation_config)
        answers = [answer.text.upper().strip().split()[0].strip() for answer in answers]
        
        responses = []
        for answer, letter2option, meta in zip(answers, letter2options, metas):
            meta = deepcopy(meta)
            race = letter2option.get(answer, None)
            if race is not None:
                race = normalize_race(race)
            meta["answer"] = race
            responses.append(meta)
        
        return responses
    

def make_prompt(include_unknown: bool = False) -> tuple[str, dict]:
    if include_unknown:
        question = "What is the race of the person in the image? Only answer if you are sure and there are visual cues in the image about the person's race or ethnicity. If you are not sure, choose the option 'race is unclear / cannot tell / no person visible'."
        options = [
            "White, European or Caucasian",
            "Black, African, or African American",
            "East Asian, such as Chinese, Japanese, or Korean",
            "Latino, Hispanic or South American",
            "Middle Eastern, Arab or North African",
            "Indian, Pakistani, or Bangladeshi",
            "Southeast Asian, such as Filipino, Thai, or Vietnamese",
            "race is unclear / cannot tell / no person visible"
        ]
        instruction = "Answer with a single letter."
    else:
        question = "What is the race of the person in the image?"
        options = ["male", "female"]
        instruction = "Answer with a single letter."

    random.shuffle(options)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter2option = {alphabet[i]: option for i, option in enumerate(options)}
    option_list = "\n".join([f"{alphabet[i]}. {option}" for i, option in enumerate(options)])
    prompt = f"{question}\n{option_list}\n\n{instruction}"
    return prompt, letter2option


def collate_fn(batch):
    images, metas = zip(*batch)
    return list(images), list(metas)


CMD_RACE_CHOICES = {
    "black": "Black / African American",
    "eastasian": "East Asian",
    "southasian": "South Asian",
    "white": "White / Caucasian",
    "middleeastern": "Middle Eastern / North African",
    "latino": "Latino / Hispanic",
    "southeastasian": "Southeast Asian",
    None: None
}


if __name__ == "__main__":
    # Set random seed
    random.seed(42)

    # Assert that we have a GPU
    assert torch.cuda.is_available(), "CUDA is not available"
    device = "cuda"

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/lustre/groups/eml/datasets/laion400m/laion400m-data")
    parser.add_argument("--model", type=str, required=True, choices=["InternVL3-2B", "InternVL3-8B", "InternVL3-14B", "deepseek-vl2", "llava-1.6", "gemma", "phi-3.5", "qwen-vl-3B", "qwen-vl-7B"])
    parser.add_argument("--race", type=str, choices=list(CMD_RACE_CHOICES.keys()), default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument('--output-path', type=str, default="results/race_labeling/")
    args = parser.parse_args()

    # Convert race
    if args.race is not None:
        cmd_race = args.race
        race = CMD_RACE_CHOICES[args.race]
    else:
        race = None

    # Make save path
    if args.start_index is not None or args.num_samples is not None:
        result_file_name = f"results_{args.start_index}_{args.num_samples}.csv"
    else:
        result_file_name = "results.csv"
    save_path = os.path.join(args.output_path, args.model)
    if race is not None:
        save_path = os.path.join(save_path, cmd_race)
    save_path = os.path.join(save_path, result_file_name)
    
    if os.path.exists(save_path):
        print(f"Results already exist at {save_path}, exiting")
        sys.exit()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load the model
    model_name = {
        "InternVL3-2B": "OpenGVLab/InternVL3-2B",
        "InternVL3-8B": "OpenGVLab/InternVL3-8B",
        "InternVL3-14B": "OpenGVLab/InternVL3-14B",
        "deepseek-vl2": "deepseek-ai/deepseek-vl2",
        "llava-1.6": "liuhaotian/llava-v1.6-vicuna-7b",
        "gemma": "google/gemma-3-4b-it",
        "phi-3.5": "microsoft/Phi-3.5-vision-instruct",
        "qwen-vl-3B": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen-vl-7B": "Qwen/Qwen2.5-VL-7B-Instruct"
    }[args.model]
    model = MLLMRaceLabeler(model_name, device, include_unknown=True)
    processor = model.get_preprocessor()

    # Load the dataset
    dataset = RaceDataset(
        args.path,
        "data/race_data/image_ids_df_with_bounding_boxes_final.csv",
        processor,
        race,
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

        if args.max_images is not None and len(results) >= args.max_images:
            break
    
    if not results:
        print("No results found, exiting")
        sys.exit()
    
    # Save the results
    if args.max_images is None:
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path, index=False)
        print(f"Saved results to {save_path}")
