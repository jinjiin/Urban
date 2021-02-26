import torch.nn as nn
import torch.nn.functional as F
from UrbanModels.lstm_raw.base import BaseModel
import torch


class GRU_Base(BaseModel):
    def __init__(self, input_feature_size=5, hidden_size=56, predict_len=1, num_layers=1):
        super().__init__()
        """
        input_size:length of feature_attribute
        """
        self.gru = nn.GRU(input_feature_size, hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.scale = nn.Linear(hidden_size, predict_len)

    def forward(self, x):
        lstm_out, _ = self.gru(x)
        output = self.scale(lstm_out[:, -1, :])
        return output


if __name__ == '__main__':
    bs = 16
    seq_len = 24
    input_size = 18

    model = GRU_Base()

    x = torch.randn(bs, seq_len, input_size)
    output = model(x)
    print(output.shape)
