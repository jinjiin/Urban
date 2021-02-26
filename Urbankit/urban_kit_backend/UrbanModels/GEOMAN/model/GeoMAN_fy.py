import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
from torch.nn import init


class GeoMAN(nn.Module):
    def __init__(self, feature_dim, hidden_state_features, num_layers_lstm):
        super(GeoMAN, self).__init__()

        # self.window_length = args.window  # window, read about it after understanding the flow of the code...What is window size? --- temporal window size (default 24 hours * 7)
        self.original_columns = feature_dim # the number of columns or features
        self.hidden_state_features = hidden_state_features
        # self.output_num = args.predict_fea
        # self.predict = data.predicted
        self.num_layers_lstm = num_layers_lstm
        self.lstm = nn.LSTM(input_size=self.original_columns+10, hidden_size=self.hidden_state_features,
                            num_layers=self.num_layers_lstm,
                            batch_first=True,
                            bidirectional=False)

        self.mlp = nn.Linear(self.hidden_state_features, 2)
        self.lc1 = nn.Linear(self.original_columns, self.original_columns)
        self.lc2 = nn.Linear(10, 10)
    def forward(self, X_local,X_global):
        #LOCAL B,T,F    GLOBAL B,T,35
        x_local = self.lc1(X_local)
        x_global = self.lc2(X_global)

        x = torch.cat([x_local,x_global],dim=2)
        lstm_hidden_states, (h_all, c_all) = self.lstm(x)
        # hn = h_all[-1].view(1, h_all.size(1), h_all.size(2))
        h_all = torch.squeeze(h_all, 0)
        # h_all = self.BN(h_all)
        output = self.mlp(h_all)

        return output
