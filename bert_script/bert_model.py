from typing import Dict

import torch
from transformers import AutoModel
from torch.nn import Embedding, GRU, Linear, Dropout, LeakyReLU, Tanh, LayerNorm, BatchNorm1d


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        num_class: int,
        seq_wise: bool = True,
    ) -> None:
        super(SeqClassifier, self).__init__()

        self.backbone = AutoModel.from_pretrained('bert-base-uncased')
        self.hidden_size = self.backbone.config.hidden_size
        self.fc = Linear(self.hidden_size, num_class)

        self.seq_wise = seq_wise

    @property
    def encoder_output_size(self) -> int:
        return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        last_hidden_states = self.backbone(batch).last_hidden_state

        if self.seq_wise:
            logits = self.fc(last_hidden_states[:, 0])
            return logits
        else:
            logits = self.fc(last_hidden_states)
            return logits
