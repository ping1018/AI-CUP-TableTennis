import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, input_channels=6, conv_out_channels=64, lstm_hidden_size=256, num_classes=2):
        super(CNN_LSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=conv_out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduces time length by half
        )
        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        # x: (batch, time_len, input_channels)
        x = x.permute(0, 2, 1)       # → (batch, input_channels, time_len)
        x = self.conv(x)             # → (batch, conv_out_channels, new_time_len)
        x = x.permute(0, 2, 1)       # → (batch, new_time_len, conv_out_channels)
        
        # Adjust lengths due to max pooling
        lengths = lengths // 2

        # LSTM
        out, _ = self.lstm(x)        # → (batch, new_time_len, hidden*2)

        # Gather the last valid output for each sequence
        batch_size = x.size(0)
        last_outputs = torch.stack([out[i, lengths[i]-1, :] for i in range(batch_size)])

        return self.fc(last_outputs)