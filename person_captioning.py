import os
import sys
import torch
import argparse
import pandas as pd
import tarfile

from tqdm import tqdm
from io import BytesIO
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from lmdeploy.vl import load_image as lmdeploy_load_image
from utils.laion_reader import Laion400mBoundingBoxDataset
from lmdeploy import pipeline, TurbomindEngineConfig, VisionConfig, GenerationConfig


os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT = "Describe the highlighted person in one continuous paragraph. Describe the context, how the person looks and what the person is doing. Focus on objective details and avoid any subjective or emotional descriptions. Only focus on the highlighted person. Be as detailed and precise as possible. Write in plain, objective and scientific language."


class Laion400mHighlightedPersonDataset(Laion400mBoundingBoxDataset):
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            bounding_boxes_by_image_keys = self.bounding_boxes_by_image_keys
        else:
            bounding_boxes_by_image_keys = self.bounding_boxes_by_image_keys[worker_info.id::worker_info.num_workers]

        for tarball in self.tarballs:
            tarball_name = os.path.basename(tarball)
            with tarfile.open(tarball, 'r') as tar:
                members = tar.getmembers()
                for member in members:
                    member_name = member.name

                    if not member_name.endswith('.jpg'):
                        continue

                    if (tarball_name, member_name) not in bounding_boxes_by_image_keys:
                        continue

                    image = tar.extractfile(member)
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')

                    bounding_boxes = self.bounding_boxes_by_image[(tarball_name, member_name)]
                    for bounding_box in bounding_boxes:
                        x1, y1, x2, y2 = bounding_box
                        image_with_bounding_box = image.copy()
                        draw = ImageDraw.Draw(image_with_bounding_box)
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                        if self.processor is not None:
                            image_with_bounding_box = self.processor(image_with_bounding_box)
                        
                        meta = {
                            "tarball_name": tarball_name,
                            "member_name": member_name,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }

                        yield image_with_bounding_box, meta


def collate_fn(batch):
    images, metas = zip(*batch)
    return list(images), list(metas)


if __name__ == "__main__":
    # Assert that we have a GPU
    assert torch.cuda.is_available(), "CUDA is not available"

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/lustre/groups/eml/datasets/laion400m/laion400m-data")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--output-path', type=str, default="results/person_captioning/")
    parser.add_argument('--model', type=str, default="OpenGVLab/InternVL3-8B")
    args = parser.parse_args()

    # Make save path
    result_file_name = f"results_{args.start_idx}_{args.num_samples}.csv"
    save_path = os.path.join(args.output_path, result_file_name)
    if os.path.exists(save_path):
        print(f"Results already exist at {save_path}, exiting")
        sys.exit()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load the model
    backend_config = TurbomindEngineConfig(session_len=8192)
    vision_config = VisionConfig(max_batch_size=args.batch_size)
    generation_config = GenerationConfig(max_new_tokens=256, do_sample=False)
    pipe = pipeline(
        args.model,
        backend_config=backend_config,
        vision_config=vision_config,
    )

    # Load the dataset
    dataset = Laion400mHighlightedPersonDataset(
        laion_400m_path=args.path,
        bounding_boxes_path="data/person_detection/bounding_boxes_filtered",
        processor=lmdeploy_load_image,
        start_idx=args.start_idx,
        num_samples=args.num_samples
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=6)

    # Run the model
    results = []
    for images, metas in tqdm(dataloader):
        prompts = [(PROMPT, image) for image in images]
        captions = pipe(prompts, gen_config=generation_config)
        captions = [caption.text for caption in captions]
        
        metas_with_captions = [
            {**meta, "caption": caption}
            for meta, caption in zip(metas, captions)
        ]
        results.extend(metas_with_captions)
    
    if not results:
        print("No results found, exiting")
        sys.exit()
    
    # Save the results
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")