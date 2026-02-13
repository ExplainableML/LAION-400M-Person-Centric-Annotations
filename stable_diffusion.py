import os
import json
import torch
import argparse

from tqdm import tqdm
from ultralytics import YOLO
from diffusers import StableDiffusionPipeline
from gender_labeling import MLLMGenderLabeler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--images-per-prompt", type=int, default=100)
    parser.add_argument("--max-generations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    args = parser.parse_args()
    
    print(f"Starting from prompt {args.start_index} with {args.num_prompts} prompts")
    
    # Load prompts
    with open("lexical_data/stable_diffusion_prompts.txt", "r") as f:
        prompts = f.read().splitlines()
        prompts = [prompt.strip() for prompt in prompts]
        prompts = prompts[args.start_index:args.start_index + args.num_prompts]
        print(f"Loaded {len(prompts)} prompts")
    
    # Load diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    
    # Load person detection model
    model = YOLO("./data/yolo/yolo11l.pt")
    model = model.to(device)
    
    # Load gender labeling model
    gender_labeler = MLLMGenderLabeler(model="OpenGVLab/InternVL3-2B", device=device, include_unknown=True)
    preprocessor = gender_labeler.get_preprocessor()

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    for prompt_index, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        model_name = args.model.split("/")[-1]
        if os.path.exists(f"results/stable_diffusion/{model_name}/images/{prompt_index + args.start_index}"):
            if len(os.listdir(f"results/stable_diffusion/{model_name}/images/{prompt_index + args.start_index}")) >= args.images_per_prompt:
                print(f"Skipping prompt {prompt_index + args.start_index} because it already has {args.images_per_prompt} images")
                continue
        
        prompt_images = []
        prompt_metas = []
        total_generations = 0
        
        while len(prompt_images) < args.images_per_prompt and total_generations < args.max_generations:
            images = pipe(prompt, num_images_per_prompt=args.batch_size).images
            total_generations += len(images)
            
            # Detect people in images
            person_detections = model(images, classes=[0], verbose=False)
        
            images_with_one_person = []
            for i, result in enumerate(person_detections):
                bounding_boxes = result.boxes.xyxy
                confidences = result.boxes.conf
            
                if len(bounding_boxes) == 1 and confidences[0].item() > 0.35:
                    x1, y1, x2, y2 = bounding_boxes[0].cpu().tolist()
                    width, height = x2 - x1, y2 - y1
                    min_sidelength = min(width, height)
                    
                    if min_sidelength > 30:
                        images_with_one_person.append((images[i], (x1, y1, x2, y2), confidences[0].item()))
        
            if len(images_with_one_person) == 0:
                continue
            
            # Label gender
            images = [preprocessor(image) for image, _, _ in images_with_one_person]
            metas = [
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": confidence}
                for _, (x1, y1, x2, y2), confidence in images_with_one_person
            ]
            gender_labeling_results = gender_labeler(images, metas)
            
            for image, meta in zip(images_with_one_person, gender_labeling_results):
                if meta["answer"] in ["male", "female"]:
                    prompt_images.append(image[0])
                    prompt_metas.append(meta)
            
            print(f"Generated {len(prompt_images)}/100 images")
        
        
        prompt_index_save = prompt_index + args.start_index
        model_name = args.model.split("/")[-1]
        os.makedirs(f"results/stable_diffusion/{model_name}/images/{prompt_index_save}", exist_ok=True)
        for j, image in enumerate(prompt_images):
            image.save(f"results/stable_diffusion/{model_name}/images/{prompt_index_save}/{j}.png")
        
        os.makedirs(f"results/stable_diffusion/{model_name}/metas/", exist_ok=True)
        with open(f"results/stable_diffusion/{model_name}/metas/{prompt_index_save}.json", "w") as f:
            json.dump(prompt_metas, f)
        
        print(f"Saved {len(prompt_images)} images for prompt {prompt_index_save}")