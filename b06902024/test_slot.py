import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Callable, List
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    idx2label: Callable,
    replace: bool,
) -> List[Dict]:

    model.eval()
    prediction = defaultdict(list)
    with torch.no_grad():
        for inputs in dataloader:
            prediction["id"] += inputs["id"]
            outputs = model(inputs["tokens_ids"].to(device))
            outputs = outputs.argmax(dim=-1)
            outputs = outputs.int().tolist()
            result = [
                [idx2label(idx) for idx in ids[1 : len(inputs["tokens"][i]) - 1]]
                for i, ids in enumerate(outputs)
            ]
            if replace:
                result = [
                    [
                        label if i == 0 or label != labels[i - 1] else label.replace("B", "I")
                        for i, label in enumerate(labels)
                    ]
                    for labels in result
                ]
            prediction["tags"] += result
    return prediction


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(
        data, vocab, tag2idx, args.max_len, is_train=False, add_labels=args.add_labels
    )
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn_slot,
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes + 3 * args.add_labels,
        seq_wise=False,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    model.to(args.device)
    # TODO: predict dataset
    prediction = predict(model, dataloader, args.device, dataset.idx2label, args.replace)
    # TODO: write prediction to file (args.pred_file)
    # df = pd.DataFrame.from_dict(prediction)
    # df.to_csv(args.pred_file, index=False, doublequote=False)
    with open(args.pred_file, "w") as f:
        f.write("id,tags\n")
        for i in range(len(dataset)):
            f.write(f'{prediction["id"][i]},{" ".join(prediction["tags"][i])}\n')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=Path, help="Path to the test file.", required=True)
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument("--ckpt_path", type=Path, help="Path to model checkpoint.", required=True)
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--add_labels", action="store_true", default=False)
    parser.add_argument("--replace", action="store_true", default=False)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
