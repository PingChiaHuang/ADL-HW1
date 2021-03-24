from typing import Dict

import torch
from torch.nn import Embedding, GRU, Linear, Sequential, Sigmoid


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
        self.gru = GRU(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.classifier = Sequential(Linear(self.encoder_output_size, num_class), Sigmoid())

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
        encoded = self.gru(embedded)
        last_encoded = encoded[:, -1, :]
        outputs = self.classifier(last_encoded)
        return outputs
        raise NotImplementedError
