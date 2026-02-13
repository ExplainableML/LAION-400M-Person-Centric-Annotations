#!/usr/bin/env python3
"""
Embed captions from a CSV file using Snowflake/snowflake-arctic-embed-m-v2.0
and save the embeddings as a PyTorch tensor.

Arguments:
  --path-to-captions: default=results/person_captions
  --start-index: required, type int
  --save-path: default=results/person_caption_embeddings
  --batch-size: default=1024 (batch size for the embeddings model)

File resolution rule:
  If start_index % 100 != 0, multiply start_index by 100.
  The script loads: {path_to_captions}/results_{start_index}_100.csv
  and reads only the column: "caption".

Output:
  Saves a tensor to {save_path}/embeddings_{start_index}_100.pt
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import List

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed captions and save as Torch tensor.")
    parser.add_argument(
        "--path-to-captions",
        default="results/person_captioning/",
        type=str,
        help="Directory containing caption CSV files.",
    )
    parser.add_argument(
        "--start-index",
        required=True,
        type=int,
        help="Start index used in the CSV filename. If not a multiple of 100, it will be multiplied by 100.",
    )
    parser.add_argument(
        "--save-path",
        default="results/person_caption_embeddings",
        type=str,
        help="Directory to save the output Torch tensor.",
    )
    parser.add_argument(
        "--batch-size",
        default=1024,
        type=int,
        help="Batch size for the embedding model.",
    )
    return parser.parse_args()


def resolve_start_index(start_index: int) -> int:
    return start_index if start_index % 100 == 0 else start_index * 100


def load_captions(csv_path: str) -> List[str]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, usecols=["caption"])  # read only the 'caption' column
    except ValueError as e:
        # This typically happens if 'caption' column is missing
        raise ValueError(f"Expected a 'caption' column in {csv_path}. Original error: {e}")

    # Drop NaNs and coerce to str (in case of mixed types)
    captions = df["caption"].dropna().astype(str).tolist()
    if len(captions) == 0:
        raise ValueError(f"No captions found in column 'caption' of {csv_path}")
    return captions


def compute_embeddings(captions: List[str], batch_size: int) -> torch.Tensor:
    model_name = "ibm-granite/granite-embedding-english-r2"

    # Choose device automatically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{model_name}' on device: {device}")

    model = SentenceTransformer(model_name, trust_remote_code=True, device=device, model_kwargs={'torch_dtype': torch.bfloat16})

    # Encode captions (documents). Do NOT pass prompt_name here (only for queries).
    print(f"Encoding {len(captions)} captions with batch_size={batch_size} …")
    embeddings: torch.Tensor = model.encode(
        captions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=False,
    )

    # Ensure tensor is on CPU before saving
    return embeddings.detach().cpu()


def main() -> None:
    args = parse_args()

    start_idx = resolve_start_index(args["start_index"] if isinstance(args, dict) else args.start_index)

    csv_filename = f"results_{start_idx}_100.csv"
    csv_path = os.path.join(args.path_to_captions, csv_filename)
    print(f"Resolved CSV path: {csv_path}")

    captions = load_captions(csv_path)

    embeddings = compute_embeddings(captions, batch_size=args.batch_size)

    # Prepare save directory and file
    os.makedirs(args.save_path, exist_ok=True)
    out_filename = f"embeddings_{start_idx}_100.pt"
    out_path = os.path.join(args.save_path, out_filename)

    torch.save(embeddings, out_path)

    print(
        f"Saved embeddings: {out_path}\n"
        f"Shape: {tuple(embeddings.shape)} (num_texts, embedding_dim)\n"
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
