import librosa
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle

def extract_mfcc_from_audio(audio_file, n_mfcc=13, max_len=800):
    """
    Extract MFCC from an audio file and pad or truncate to a fixed length.
    :param audio_file: Path to the audio file.
    :param n_mfcc: Number of MFCC features to extract.
    :param max_len: Maximum length of the MFCC sequence (padding/truncation).
    :return: Numpy array of MFCCs.
    """
    y, sr = librosa.load(audio_file, sr=8000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    if mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]
    else:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    
    return mfcc

def preprocess_data(data_dir, save_dir, n_mfcc=13, max_len=800):
    """
    Preprocess the audio data and save the MFCCs and labels as numpy arrays.
    :param data_dir: Directory containing the raw audio files (0-9).
    :param save_dir: Directory where processed data will be saved.
    :param n_mfcc: Number of MFCCs to extract.
    :param max_len: Maximum length of MFCC sequences.
    """
    X = []
    y = []
    
    label_encoder = LabelEncoder()
    label_encoder.fit([str(i) for i in range(10)])
    
    # Iterate through each folder (0-9)
    for label in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label)
        
        if os.path.isdir(folder_path):
            for file in tqdm(os.listdir(folder_path)):
                if file.endswith(".wav"):
                    audio_file = os.path.join(folder_path, file)
                    
                    mfcc = extract_mfcc_from_audio(audio_file, n_mfcc, max_len)
                    X.append(mfcc)
                    y.append(label)
    
    X = np.array(X)
    y = np.array(label_encoder.transform(y))
    
    np.save(os.path.join(save_dir, 'X.npy'), X)
    np.save(os.path.join(save_dir, 'y.npy'), y)
    
    
    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"Data preprocessing complete! Saved to {save_dir}")

