from typing import Dict

import torch
from torch.nn import Embedding, GRU, Linear, Dropout, LeakyReLU, Tanh, LayerNorm, BatchNorm1d


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        embedding_size = embeddings.size(1)
        self.layernorm = LayerNorm(embedding_size)
        self.gru = GRU(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.leakyrelu = LeakyReLU()
        self.tanh = Tanh()
        self.dropout = Dropout(dropout)
        output_size = hidden_size * (1 + bidirectional)
        self.batchnorm = BatchNorm1d(output_size)
        self.fc = Linear(output_size, num_class)

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        output_size = self.hidden_size * (1 + self.bidirectional)
        return output_size
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embedded = self.embed(batch)
        encoded, _ = self.gru(embedded)
        encoded = self.dropout(encoded)
        # last_encoded = torch.cat([encoded[:, -1, :self.hidden_size], encoded[:, 0, self.hidden_size:]], dim=-1)
        last_encoded = self.batchnorm(encoded[:, -1])
        logits = self.fc(last_encoded)
        return logits
        raise NotImplementedError
