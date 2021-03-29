from typing import List, Dict
import torch
from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        is_train: bool = True,
        add_labels: bool = True,
        ignore: bool = False,
        replace: bool = False,
    ):
        self.data = data
        self.vocab = vocab
        if add_labels:
            label_mapping[vocab.PAD] = len(label_mapping)
            label_mapping[vocab.BOS] = len(label_mapping) + 1
            label_mapping[vocab.EOS] = len(label_mapping) + 2
        self.label_mapping = label_mapping
        self._idx2label = {idx: label for label, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.is_train = is_train
        self.add_labels = add_labels
        self.ignore = ignore
        self.replace = replace

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn_intent(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        batched_samples = {key: [sample[key] for sample in samples] for key in samples[0]}
        batched_samples["text"] = [
            ("[BOS] " + text + " [EOS]").split() for text in batched_samples["text"]
        ]
        batched_samples["text_ids"] = torch.LongTensor(
            self.vocab.encode_batch(batched_samples["text"])
        )
        if self.is_train:
            batched_samples["intent_ids"] = torch.LongTensor(
                [self.label2idx(label) for label in batched_samples["intent"]]
            )
        return batched_samples
        raise NotImplementedError

    def collate_fn_slot(self, samples: List[Dict]) -> Dict:
        batched_samples = {key: [sample[key] for sample in samples] for key in samples[0]}
        batched_samples["tokens"] = [
            ["[BOS]", *tokens, "[EOS]"] for tokens in batched_samples["tokens"]
        ]
        batched_samples["tokens_ids"] = torch.LongTensor(
            self.vocab.encode_batch(batched_samples["tokens"])
        )
        if self.is_train:
            if self.replace:
                batched_samples["tags"] = [
                    [tag.replace("I", "B") for tag in tags] for tags in batched_samples["tags"]
                ]
            batched_samples["tags_ids"] = torch.full_like(
                batched_samples["tokens_ids"],
                self.label2idx("[PAD]")
                if self.add_labels
                else -100
                if self.ignore
                else self.label2idx("O"),
            )
            for i, tags in enumerate(batched_samples["tags"]):
                batched_samples["tags_ids"][i, 1 : 1 + len(tags)] = torch.LongTensor(
                    [self.label2idx(label) for label in tags]
                )
                if self.add_labels:
                    batched_samples["tags_ids"][i, 1 + len(tags)] = self.label2idx("[EOS]")
            if self.add_labels:
                batched_samples["tags_ids"][:, 0] = self.label2idx("[BOS]")
            # print(batched_samples["tags"][:10])
            # print(batched_samples["tags_ids"][:10])
            # exit()
        return batched_samples

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
