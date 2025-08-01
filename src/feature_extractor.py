import librosa
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle

label_map = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
}

def add_noise(y, noise_factor=0.01):
    """ Add random noise to the audio signal """
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def extract_mfcc_from_audio(audio_file, n_mfcc=13, max_len=800, noise_factor=0.01):
    """
    Extract MFCC from an audio file and pad or truncate to a fixed length.
    :param audio_file: Path to the audio file.
    :param n_mfcc: Number of MFCC features to extract.
    :param max_len: Maximum length of the MFCC sequence (padding/truncation).
    :param noise_factor: Factor for adding noise to audio.
    :return: Numpy array of MFCCs.
    """
    y, sr = librosa.load(audio_file, sr=8000)
    y = add_noise(y, noise_factor)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    if mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]
    else:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    
    scaler = StandardScaler()
    mfcc = scaler.fit_transform(mfcc.T).T 
    return mfcc

def preprocess_data(data_dir, save_dir, n_mfcc=13, max_len=800, noise_factor=0.01):
    """
    Preprocess the audio data and save the MFCCs and labels as numpy arrays.
    :param data_dir: Directory containing the raw audio files (0-9).
    :param save_dir: Directory where processed data will be saved.
    :param n_mfcc: Number of MFCCs to extract.
    :param max_len: Maximum length of MFCC sequences.
    :param noise_factor: Noise factor for augmentation.
    """
    X = []
    y = []
    
    for label in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label)
        
        if label not in label_map:
            print(f"Skipping folder with invalid label: {label}")
            continue
        
        if os.path.isdir(folder_path):
            for file in tqdm(os.listdir(folder_path)):
                if file.endswith(".wav"): 
                    audio_file = os.path.join(folder_path, file)
                    print(f"Processing file: {audio_file}") 
                    
                    mfcc = extract_mfcc_from_audio(audio_file, n_mfcc, max_len, noise_factor)
                    X.append(mfcc)
                    y.append(label_map[label]) 
    
    X = np.array(X)
    y = np.array(y)  

    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    
    np.save(os.path.join(save_dir, 'X.npy'), X)
    np.save(os.path.join(save_dir, 'y.npy'), y)
    
    with open(os.path.join(save_dir, 'label_map.pkl'), 'wb') as f:
        pickle.dump(label_map, f)
    
    print(f"Data preprocessing complete! Saved to {save_dir}")