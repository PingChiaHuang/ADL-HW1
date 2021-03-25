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
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
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

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
