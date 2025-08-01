import torch
import torch.nn as nn
import torch.optim as optim

class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SpeechRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output

def create_model(input_size, hidden_size, num_classes):
    return SpeechRecognitionModel(input_size, hidden_size, num_classes)
