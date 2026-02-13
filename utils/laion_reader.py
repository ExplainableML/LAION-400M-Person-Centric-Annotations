import os
import torch
import tarfile
import pandas as pd

from tqdm import tqdm
from PIL import Image
from io import BytesIO
from collections import defaultdict
from torch.utils.data import IterableDataset


class Laion400mImageDataset(IterableDataset):
    def __init__(self, laion_400m_path: str, transform = None, start_idx: int = None, num_samples: int = None) -> None:
        self.laion_400m_path = laion_400m_path
        self.transform = transform
        self.start_idx = start_idx
        self.num_samples = num_samples

         # Get the tarball ids
        self.tarballs = [tarball for tarball in os.listdir(laion_400m_path) if tarball.endswith('.tar')]
        self.tarballs = list(sorted(self.tarballs, key=lambda x: int(x.split('.')[0])))
        self.tarballs = [os.path.join(laion_400m_path, tarball) for tarball in self.tarballs]

        if start_idx is not None:
            self.tarballs = self.tarballs[start_idx:]
        if num_samples is not None:
            self.tarballs = self.tarballs[:num_samples]

        print(f"Loaded {len(self.tarballs)} tarballs")
        print(self.tarballs)
    
    def __iter__(self):
        for tarball in self.tarballs:
            tarball_name = os.path.basename(tarball)

            with tarfile.open(tarball, 'r') as tar:
                members = tar.getmembers()
                for member in members:
                    member_name = member.name

                    if not member_name.endswith('.jpg'):
                        continue

                    image = tar.extractfile(member)
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')

                    if self.transform is not None:
                        image = self.transform(image)

                    yield tarball_name, member_name, image
    
    def collate_fn(self, batch):
        tarball_names, member_names, images = zip(*batch)
        return list(tarball_names), list(member_names), list(images)


class Laion400mBoundingBoxDataset(IterableDataset):
    def __init__(self, laion_400m_path: str, bounding_boxes_path: str, processor = None, start_idx: int = None, num_samples: int = None) -> None:
        self.laion_400m_path = laion_400m_path
        self.bounding_boxes_path = bounding_boxes_path
        self.processor = processor
        self.start_idx = start_idx
        self.num_samples = num_samples

        # Get the tarball ids
        self.tarballs = [tarball for tarball in os.listdir(laion_400m_path) if tarball.endswith('.tar')]
        self.tarballs = list(sorted(self.tarballs, key=lambda x: int(x.split('.')[0])))
        self.tarballs = [os.path.join(laion_400m_path, tarball) for tarball in self.tarballs]

        if start_idx is not None:
            self.tarballs = self.tarballs[start_idx:]
        if num_samples is not None:
            self.tarballs = self.tarballs[:num_samples]

        print(f"Loaded {len(self.tarballs)} tarballs")
        print(self.tarballs)

        # Get the bounding boxes
        self.bounding_boxes_by_image = defaultdict(list)
        for tarball in self.tarballs:
            tarball_name = os.path.basename(tarball)
            bounding_boxes_file = os.path.join(bounding_boxes_path, f"{tarball_name.replace('.tar', '')}.csv")
            bounding_boxes = pd.read_csv(bounding_boxes_file)
            for _, row in tqdm(bounding_boxes.iterrows()):
                member_name = row["member_name"]
                x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
                self.bounding_boxes_by_image[(tarball_name, member_name)].append((x1, y1, x2, y2))
        
        self.bounding_boxes_by_image_keys = list(self.bounding_boxes_by_image.keys())
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            bounding_boxes_by_image_keys = self.bounding_boxes_by_image_keys
        else:
            bounding_boxes_by_image_keys = self.bounding_boxes_by_image_keys[worker_info.id::worker_info.num_workers]

        bounding_boxes_by_image_keys = set(bounding_boxes_by_image_keys)

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
                        image_cropped = image.crop((x1, y1, x2, y2))

                        if self.processor is not None:
                            image_cropped = self.processor(image_cropped)

                        meta = {
                            "tarball_name": tarball_name,
                            "member_name": member_name,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }

                        yield image_cropped, meta
        
        
        
