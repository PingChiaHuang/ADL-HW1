import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, lr_scheduler
from torch.nn import CrossEntropyLoss

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def init_weights(self):
    for m in self.modules():
        if type(m) in [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for ih in param.chunk(3, 0):
                        torch.nn.init.xavier_uniform_(ih)
                elif "weight_hh" in name:
                    for hh in param.chunk(3, 0):
                        torch.nn.init.orthogonal_(hh)
                # elif "bias" in name:
                #     param.data.fill_(0)


def run_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim,
    lr_scheduler: torch.optim.lr_scheduler,
    mode: str,
    total_len: int,
    batch_size: int,
) -> Dict:

    is_train = mode == TRAIN
    model.train() if is_train else model.eval()

    num_batch = int(np.ceil(total_len / batch_size))
    criterion = CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    for i, inputs in enumerate(dataloader, 1):
        with torch.set_grad_enabled(is_train):
            outputs = model(inputs["tokens_ids"].to(device))
            labels = inputs["tags_ids"].to(device)
            optimizer.zero_grad()
            loss = criterion(outputs.permute(0, 2, 1), labels)

            if is_train:
                loss.backward()
                optimizer.step()

            outputs = outputs.argmax(dim=-1)
            loss = loss.item()
            correct = torch.all((outputs == labels) | (labels == -100), dim=-1).float()
            print(
                f"{i:03d}/{num_batch} [{mode.center(5)}] Loss: {loss:.4f} ACC: {correct.mean().item():.4f}",
                end="\r",
            )

            total_loss += loss * labels.size(0)
            total_correct += correct.sum().item()

    if not is_train:
        lr_scheduler.step(total_loss)

    result = {}
    result["loss"] = total_loss / total_len
    result["acc"] = total_correct / total_len
    return result


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    dataloaders = {
        split: DataLoader(
            split_dataset,
            batch_size=args.batch_size,
            shuffle=(split == TRAIN),
            num_workers=args.num_workers,
            collate_fn=split_dataset.collate_fn_slot,
        )
        for split, split_dataset in datasets.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        datasets[TRAIN].num_classes,
        seq_wise=False,
    )
    model.apply(init_weights)
    # TODO: init optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedular = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    # epoch_pbar = trange(args.num_epoch, desc="Epoch")
    model.to(args.device)
    best_loss = np.inf
    for epoch in range(1, args.num_epoch + 1):
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        for split in [TRAIN, DEV]:
            result = run_one_epoch(
                model,
                dataloaders[split],
                device=args.device,
                optimizer=optimizer,
                lr_scheduler=schedular,
                mode=split,
                total_len=len(datasets[split]),
                batch_size=args.batch_size,
            )
            print(
                f"{epoch:03d}/{args.num_epoch} [{split.center(5)}] Loss: {result['loss']:.4f} Acc: {result['acc']:.4f}"
            )

            if split == DEV:
                if best_loss > result["loss"]:
                    best_loss = result["loss"]
                    torch.save(
                        model.state_dict(),
                        args.ckpt_dir / f"{epoch:03d}-{result['loss']:.4f}-{result['acc']:.4f}.pt",
                    )

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
