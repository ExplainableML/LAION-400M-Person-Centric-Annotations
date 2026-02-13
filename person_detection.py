import os
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from ultralytics import YOLO
from torch.utils.data import DataLoader
from utils.laion_reader import Laion400mImageDataset


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/lustre/groups/eml/datasets/laion400m/laion400m-data")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--output-path', type=str, default="results/person_detection/")
    args = parser.parse_args()

    # Determine the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the YOLO model
    model = YOLO("./data/yolo/yolo11l.pt")
    model = model.to(device)

    # Load the dataset
    dataset = Laion400mImageDataset(args.path, start_idx=args.start_idx, num_samples=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    if len(dataset.tarballs) == 0:
        print("No tarballs found in the dataset")
        exit()

    # Prepare directory for saving sample predictions
    os.makedirs(os.path.join(args.output_path, "sample_predictions"), exist_ok=True)
    sample_prediction_dir = os.path.join(args.output_path, "sample_predictions")
    sample_predictions_id = 0

    # Iterate over the dataset
    all_detections = []
    for tarball_names, member_names, images in tqdm(dataloader):
        results = model(images, classes=[0], verbose=False)

        for i, (image, result) in enumerate(zip(images, results)):
            tarball_name = tarball_names[i]
            member_name = member_names[i]
            bounding_boxes = result.boxes.xyxy
            confidences = result.boxes.conf

            for bounding_box, confidence in zip(bounding_boxes, confidences):
                x1, y1, x2, y2 = bounding_box.cpu().tolist()
                confidence = confidence.item()
                # Calculate the area of the bounding box
                area = (x2 - x1) * (y2 - y1)
                # Calculate the area of the image covered by the bounding box
                image_area = image.height * image.width
                # Calculate the percentage of the image covered by the bounding box
                percentage = area / image_area

                all_detections.append(
                    {
                        "tarball_name": tarball_name,
                        "member_name": member_name,
                        "percentage": percentage,
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
    
    # Convert the list of detections to a pandas DataFrame
    df = pd.DataFrame(all_detections)
    # Save the DataFrame to a CSV file
    os.makedirs(args.output_path, exist_ok=True)
    save_path = os.path.join(args.output_path, f"detections_{args.start_idx}_{args.num_samples}.csv")
    df.to_csv(save_path, index=False)
