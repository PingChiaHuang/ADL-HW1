from typing import List, Dict
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        label_mapping: Dict[str, int],
        max_len: int,
        is_train: bool = True,
    ):
        self.data = data
        self.label_mapping = label_mapping
        self._idx2label = {idx: label for label, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.is_train = is_train
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
        batched_samples["text_ids"] = torch.LongTensor(
            self.tokenizer(batched_samples["text"], padding=True).input_ids
        )
        if self.is_train:
            batched_samples["intent_ids"] = torch.LongTensor(
                [self.label2idx(label) for label in batched_samples["intent"]]
            )
        return batched_samples

    def collate_fn_slot(self, samples: List[Dict]) -> Dict:
        batched_samples = {key: [sample[key] for sample in samples] for key in samples[0]}
        batched_samples["tokens_ids"] = torch.LongTensor(
            self.tokenizer(batched_samples["tokens"], padding=True, is_split_into_words=True).input_ids
        )
        if self.is_train:
            batched_samples["tags_ids"] = torch.full_like(
                batched_samples["tokens_ids"],
                self.label2idx("O"),
            )
            for i, tags in enumerate(batched_samples["tags"]):
                batched_samples["tags_ids"][i, 1 : 1 + len(tags)] = torch.LongTensor(
                    [self.label2idx(label) for label in tags]
                )
        return batched_samples

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
