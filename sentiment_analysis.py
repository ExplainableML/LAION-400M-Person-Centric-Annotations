import os
import sys
import torch
import argparse
import pandas as pd

import time
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict
from torch.utils.data import DataLoader
from pysentimiento import create_analyzer
from torch.utils.data import Dataset as TorchDataset
from datasets.utils.logging import disable_progress_bar
from pysentimiento.preprocessing import preprocess_tweet
from nltk.sentiment.vader import SentimentIntensityAnalyzer

disable_progress_bar()


class TextDataset(TorchDataset):
    def __init__(self, text_paths: list[str], tokenizer):
        self.text_paths = text_paths
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.text_paths)
    
    def __getitem__(self, idx: int) -> pd.DataFrame:
        text_path = self.text_paths[idx]
        texts = pd.read_parquet(text_path)

        if self.tokenizer is not None:
            texts_list = texts["text"].tolist()
            data = {
                "text": [
                    preprocess_tweet(sent) for sent in texts_list
                ]
            }
            dataset = Dataset.from_dict(data)
            dataset = dataset.map(self._tokenize, batched=True, batch_size=32, num_proc=1)
            return texts, dataset
        else:
            return texts, None
    
    def _tokenize(self, batch: dict) -> dict:
        return self.tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
    
    def collate_fn(self, batch: list[pd.DataFrame]) -> pd.DataFrame:
        assert len(batch) == 1
        texts, dataset = batch[0]
        return texts, dataset


def hate_speech_scores(texts: pd.DataFrame, dataset: Dataset, analyzer) -> pd.DataFrame:
    output = analyzer.eval_trainer.predict(dataset)
    logits = torch.tensor(output.predictions)

    scores = [
        analyzer._get_output(sent, logits_row.view(1, -1), context=None).probas
        for sent, logits_row in zip(dataset["text"], logits)
    ]

    # Transpose scores
    scores_transposed = defaultdict(list)
    for score in scores:
        for key, value in score.items():
            scores_transposed[key].append(value)

    # Add scores to dataframe
    for key, value in scores_transposed.items():
        texts[key] = value
    
    return texts


def sentiment_scores(texts: pd.DataFrame, dataset: Dataset, analyzer) -> pd.DataFrame:
    output = analyzer.eval_trainer.predict(dataset)
    logits = torch.tensor(output.predictions)

    scores = [
        analyzer._get_output(sent, logits_row.view(1, -1), context=None).probas
        for sent, logits_row in zip(dataset["text"], logits)
    ]

    # Transpose scores
    scores_transposed = defaultdict(list)
    for score in scores:
        for key, value in score.items():
            scores_transposed[key].append(value)

    # Add scores to dataframe
    for key, value in scores_transposed.items():
        texts[key] = value
    
    return texts


def vader_scores(texts: pd.DataFrame) -> pd.DataFrame:
    texts_as_list = texts["text"].tolist()
    scores_transposed = defaultdict(list)
    sid = SentimentIntensityAnalyzer()
    for text in texts_as_list:
        sentiment_dict = sid.polarity_scores(text)
        for key, value in sentiment_dict.items():
            scores_transposed[key].append(value)
    
    for key, value in scores_transposed.items():
        texts[key] = value
    
    return texts


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/text_index/laion400m_text_index.parquet/")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-path", type=str, default="results/sentiment_analysis/")
    parser.add_argument("--statistic", required=True, type=str, choices=["hate_speech", "vader", "sentiment"])
    args = parser.parse_args()

    start_time = time.time()

    # Load the data
    assert isinstance(args.start_idx, int) and isinstance(args.num_samples, int)
    assert args.start_idx >= 0 and args.num_samples > 0 and args.num_samples is not None

    tarball_indices = [str(i).zfill(5) for i in range(args.start_idx, args.start_idx + args.num_samples)]
    text_paths = [os.path.join(args.data_path, f"tarball_index={tarball_index}") for tarball_index in tarball_indices]
    text_paths = [path for path in text_paths if os.path.exists(path)]

    result_path = os.path.join(args.output_path, args.statistic)
    os.makedirs(result_path, exist_ok=True)
    result_file_name = f"results_{args.start_idx}_{args.num_samples}.csv"
    if os.path.exists(os.path.join(result_path, result_file_name)):
        print(f"Results already exist at {os.path.join(result_path, result_file_name)}, exiting")
        sys.exit()

    # Load the analyzer (if needed)
    if args.statistic == "hate_speech":
        analyzer = create_analyzer(task="hate_speech", lang="en")
        tokenizer = analyzer.tokenizer
    elif args.statistic == "sentiment":
        analyzer = create_analyzer(task="sentiment", lang="en")
        tokenizer = analyzer.tokenizer
    elif args.statistic == "vader":
        tokenizer = None
    else:
        raise ValueError(f"Invalid statistic: {args.statistic}")

    dataset = TextDataset(text_paths, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn, num_workers=10)

    # Process the data
    results = []
    for texts, dataset in tqdm(dataloader):
        if args.statistic == "hate_speech":
            texts_with_scores = hate_speech_scores(texts, dataset, analyzer)
        elif args.statistic == "sentiment":
            texts_with_scores = sentiment_scores(texts, dataset, analyzer)
        elif args.statistic == "vader":
            texts_with_scores = vader_scores(texts)
        else:
            raise ValueError(f"Invalid statistic: {args.statistic}")
        results.append(texts_with_scores)

    # Save the results
    results_df = pd.concat(results)
    results_df.to_csv(os.path.join(result_path, result_file_name), index=False)

    end_time = time.time()
    print(f"Saved results to {os.path.join(result_path, result_file_name)}")
    print(f"Time taken: {end_time - start_time} seconds")

