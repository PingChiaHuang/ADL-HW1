import json
import logging
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from random import seed

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(args):
    seed(args.rand_seed)

    tags = set()
    for split in ["train", "eval"]:
        dataset_path = args.data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

        tags.update({tag for instance in dataset for tag in instance["tags"]})

    # tags = {tag for tag in tags if "I" not in tag}
    tag2idx = {tag: i for i, tag in enumerate(sorted(tags))}
    tag_idx_path = args.output_dir / "tag2idx.json"
    tag_idx_path.write_text(json.dumps(tag2idx, indent=2))
    logging.info(f"Tag 2 index saved at {str(tag_idx_path.resolve())}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument("--rand_seed", type=int, help="Random seed.", default=13)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./bert_cache/slot/",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
