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
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device, idx2label: Callable
) -> List[Dict]:

    model.eval()
    prediction = defaultdict(list)
    with torch.no_grad():
        for inputs in dataloader:
            prediction["id"] += inputs["id"]
            outputs = model(inputs["text_ids"].to(device))
            outputs = torch.argmax(outputs, dim=-1)
            outputs = outputs.int().tolist()
            prediction["intent"] += [idx2label(idx) for idx in outputs]
    return prediction


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len, is_train=False)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn_intent,
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    model.to(args.device)
    # TODO: predict dataset
    prediction = predict(model, dataloader, args.device, dataset.idx2label)
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as f:
        print('id,intent', file=f)
        for ids, intents in zip(prediction['id'], prediction['intent']):
            print(f'{ids},{intents}', file=f)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=Path, help="Path to the test file.", required=True)
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument("--ckpt_path", type=Path, help="Path to model checkpoint.", required=True)
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
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
