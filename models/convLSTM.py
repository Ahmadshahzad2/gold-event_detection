from torch import nn
import torch

class ConvLSTMModel(nn.Module):
    def __init__(self, input_shape, hidden_dim, num_classes=1):
        super(ConvLSTMModel, self).__init__()
        self.segment_length, self.keypoints, self.coords = input_shape
        
        self.conv1d = nn.Conv1d(in_channels=self.keypoints * self.coords, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        # Change to output a single value for binary classification
        self.fc = nn.Linear(hidden_dim, num_classes)  # Only one output neuron for binary classification
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), self.keypoints * self.coords, self.segment_length)
        x = self.conv1d(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)
        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.activation(x)  # Apply chosen activation
        return x