import torch.nn as nn
from transformers import AutoModel

class EmbeddingModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", freeze=True, pooling="mean"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            return last_hidden[:, 0, :]

        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / token_counts