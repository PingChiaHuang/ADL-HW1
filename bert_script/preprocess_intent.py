import json
import logging
import pickle
import re
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from random import random, seed
from typing import List, Dict

import torch
from tqdm.auto import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):
    seed(args.rand_seed)

    intents = set()
    for split in ["train", "eval"]:
        dataset_path = args.data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

        intents.update({instance["intent"] for instance in dataset})

    intent2idx = {tag: i for i, tag in enumerate(intents)}
    intent_tag_path = args.output_dir / "intent2idx.json"
    intent_tag_path.write_text(json.dumps(intent2idx, indent=2))
    logging.info(f"Intent 2 index saved at {str(intent_tag_path.resolve())}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument("--rand_seed", type=int, help="Random seed.", default=13)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./bert_cache/intent/",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
