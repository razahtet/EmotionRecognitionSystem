import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import librosa
from datasets import load_dataset, Audio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
EMOTION_MAP = {
    "HAP": 0,  # Happy
    "ANG": 1,  # Angry
    "SAD": 2,  # Sad
    "NEU": 3   # Neutral
}
epoch_num = 50

# CNN Model Architecture (Deliverable #3)
class EmotionCNN(nn.Module):
    def __init__(self, num_emotions=4):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_emotions)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Feature Extraction (Deliverable #1)
class AudioProcessor:
    def __init__(self):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=N_MELS, 
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        self.db_transform = T.AmplitudeToDB()

    def process(self, audio_array):
        audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
        mel_spec = self.mel_transform(audio_tensor)
        mel_spec_db = self.db_transform(mel_spec)
        # Resize to 128x128 for consistent CNN input
        input_tensor = torch.nn.functional.interpolate(
            mel_spec_db.unsqueeze(0), size=(128, 128)
        )
        return input_tensor.squeeze(0) # Shape: (1, 128, 128)

# Data Loading & Pipeline (Deliverable #4)
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]['spectrogram'], self.data[idx]['label']

def prepare_data(dataset, processor):
    processed = []
    print("Processing audio into Mel-Spectrograms...")
    for i, sample in enumerate(dataset['train']):
        # Extract emotion label from filename structure
        file_path = sample['audio']['path']
        emotion_code = file_path.split('_')[2] 
        
        if emotion_code in EMOTION_MAP:
            spec = processor.process(sample['audio']['array'])
            processed.append({
                'spectrogram': spec,
                'label': EMOTION_MAP[emotion_code]
            })
        if (i + 1) % 500 == 0:
            print(f"  Done: {i+1} samples")
    return processed

# Training Loop
def train_model(train_loader, val_loader, device):
    model = EmotionCNN(num_emotions=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nStarting Training on {device}...")
    
    for epoch in range(epoch_num):
        model.train()
        # Progress Bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoch_num} [Train]")
        
        for batch_spec, batch_label in train_bar:
            batch_spec, batch_label = batch_spec.to(device), batch_label.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_spec)
            loss = criterion(outputs, batch_label)
            loss.backward()
            optimizer.step()
            
            # Update progress bar with current loss
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Validation Progress Bar
        model.eval()
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epoch_num} [Val]", leave=False)
        with torch.no_grad():
            for batch_spec, batch_label in val_bar:
                batch_spec, batch_label = batch_spec.to(device), batch_label.to(device)
                outputs = model(batch_spec)
        
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load from your cleaned repository
    raw_ds = load_dataset("razahtet/crema-d-cleaned")

    raw_ds = raw_ds.cast_column("audio", Audio(sampling_rate=16000))
    
    processor = AudioProcessor()
    processed_data = prepare_data(raw_ds, processor)
    
    # Split 80/20
    split = int(len(processed_data) * 0.8)
    train_ds = AudioDataset(processed_data[:split])
    val_ds = AudioDataset(processed_data[split:])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    model = train_model(train_loader, val_loader, device)
    torch.save(model.state_dict(), f"{epoch_num}_epochs_emotion_cnn.pth")
    print(f"\nSuccess: Model trained and saved as {epoch_num}_epochs_emotion_cnn.pth")