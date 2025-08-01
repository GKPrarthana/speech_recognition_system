import torch
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import create_model 
def extract_mfcc_from_audio(audio_file, n_mfcc=13, max_len=800):
    y, sr = librosa.load(audio_file, sr=8000) 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc) 
    
    if mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]
    else:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    
    scaler = StandardScaler()
    mfcc = scaler.fit_transform(mfcc.T).T  
    return mfcc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(input_size=13, hidden_size=256, num_classes=10).to(device) 
model.load_state_dict(torch.load('../models/speech_recognition_best.pth', map_location=torch.device('cpu')))
model.eval() 

def predict_digit(audio_file):
    mfcc = extract_mfcc_from_audio(audio_file)  
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)  
    
    with torch.no_grad(): 
        output = model(mfcc_tensor) 
        _, predicted = torch.max(output, 1)  
        return predicted.item()  

audio_file = '../recordings/two.wav' 
predicted_digit = predict_digit(audio_file)  
print(f"Predicted digit: {predicted_digit}") 
