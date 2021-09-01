import torch
import torch.nn as nn

import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, args, input_dim=128, num_labels=2):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(args.dropout_rate)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)   # [batch_size, hidden_dim]

        return self.linear(x)