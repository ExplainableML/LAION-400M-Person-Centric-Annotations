import os
import torch
import logging
import argparse
import numpy as np

from torch.utils.data import DataLoader, IterableDataset
from utils.dictionary_learning.training import trainSAE
from utils.dictionary_learning.trainers import StandardTrainer, JumpReluTrainer, BatchTopKTrainer, TopKTrainer, MatryoshkaBatchTopKTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sparse_autoencoder_trainers = {
    "StandardTrainer": StandardTrainer,
    "JumpReluTrainer": JumpReluTrainer,
    "BatchTopKTrainer": BatchTopKTrainer,
    "TopKTrainer": TopKTrainer,
    "MatryoshkaBatchTopKTrainer": MatryoshkaBatchTopKTrainer,
}

class EmbeddingDataset(IterableDataset):
    def __init__(self, path_to_embeddings: str = "results/person_caption_embeddings/"):
        self.path_to_embeddings = path_to_embeddings
        self.embedding_files = [f for f in os.listdir(self.path_to_embeddings) if f.endswith(".pt")]

    def __iter__(self):
        while True:
            for file in self.embedding_files:
                embeddings = torch.load(os.path.join(self.path_to_embeddings, file))
                for embedding in embeddings:
                    yield embedding


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--trainer", type=str, default='StandardTrainer')
    parser.add_argument("--expansion-factor", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-path", type=str, default='results/sae/trained_models/')
    parser.add_argument("--save-interval", type=int, default=100)
    args = parser.parse_args()

    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = EmbeddingDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, collate_fn=torch.stack, drop_last=False)

    # Train SAE
    steps = args.steps
    warmup_steps = args.warmup_ratio * steps
    save_steps = list(range(0, steps, args.save_interval)) + [steps-1]
    trainer = sparse_autoencoder_trainers[args.trainer]

    trainer_cfg = {
        "trainer": trainer,
        "activation_dim": args.embedding_dim,
        "dict_size": args.expansion_factor * args.embedding_dim,
        "lr": args.lr,
        "device": device,
        "steps": steps,
        "lm_name": "granite",
        "layer": "embedding",
        "warmup_steps": warmup_steps,
    }

    if args.trainer in ["StandardTrainer", "JumpReluTrainer"]:
        trainer_cfg["sparsity_warmup_steps"] = warmup_steps
    elif args.trainer in ["BatchTopKTrainer", "TopKTrainer", "MatryoshkaBatchTopKTrainer"]:
        trainer_cfg["k"] = args.top_k
    
    if args.trainer == "MatryoshkaBatchTopKTrainer":
        dict_size = args.expansion_factor * args.embedding_dim
        group_fractions = [2 ** (i+1) for i in range(int(np.log2(dict_size)))]
        group_fractions = [size / dict_size for size in group_fractions]
        group_fractions[-1] = 1 - sum(group_fractions[:-1])
        trainer_cfg["group_fractions"] = group_fractions

    trainSAE(data=dataloader, trainer_configs=[trainer_cfg], steps=steps, save_steps=save_steps, save_dir=args.save_path, verbose=True, log_steps=100)