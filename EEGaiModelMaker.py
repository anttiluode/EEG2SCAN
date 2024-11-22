import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mne
from scipy.signal import butter, lfilter
from tqdm import tqdm
import logging
import gradio as gr
import tempfile

# ===============================
# 1. Configuration and Parameters
# ===============================

class Config:
    def __init__(self):
        self.fs = 100.0  # Sampling frequency (Hz)
        self.epoch_length = 1  # Epoch length in seconds
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 49)
        }
        self.latent_dim = 64
        self.eeg_batch_size = 64
        self.eeg_epochs = 50
        self.learning_rate = 1e-3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================
# 2. Data Preprocessing
# ===============================

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    high = min(high, 0.99)

    if low >= high:
        raise ValueError(f"Invalid band: lowcut={lowcut}Hz, highcut={highcut}Hz.")

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data, axis=1)
    return y

def preprocess_eeg(raw, config):
    """Preprocess EEG data by filtering into frequency bands and epoching."""
    fs = raw.info['sfreq']
    channels = raw.info['nchan']
    samples_per_epoch = int(config.epoch_length * fs)
    num_epochs = raw.n_times // samples_per_epoch

    processed_data = []

    for epoch in tqdm(range(num_epochs), desc="Preprocessing EEG Epochs"):
        start_sample = epoch * samples_per_epoch
        end_sample = start_sample + samples_per_epoch
        epoch_data = raw.get_data(start=start_sample, stop=end_sample)

        band_powers = []
        for band_name, band in config.frequency_bands.items():
            try:
                filtered = bandpass_filter(epoch_data, band[0], band[1], fs)
                power = np.mean(filtered ** 2, axis=1)
                band_powers.append(power)
            except ValueError as ve:
                logging.error(f"Skipping band {band_name} for epoch {epoch+1}: {ve}")
                band_powers.append(np.zeros(channels))

        band_powers = np.stack(band_powers, axis=1)
        processed_data.append(band_powers)

    processed_data = np.array(processed_data)
    processed_data = np.transpose(processed_data, (0, 2, 1))

    epochs_mean = np.mean(processed_data, axis=(0, 1), keepdims=True)
    epochs_std = np.std(processed_data, axis=(0, 1), keepdims=True)
    epochs_normalized = (processed_data - epochs_mean) / epochs_std

    return epochs_normalized

# ===============================
# 3. Dataset Class
# ===============================

class EEGDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

# ===============================
# 4. Model Definition
# ===============================

class EEGAutoencoder(nn.Module):
    def __init__(self, channels=5, frequency_bands=7, latent_dim=64):
        super(EEGAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2))
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * channels * (frequency_bands // 4), latent_dim)
        self.fc2 = nn.Linear(latent_dim, 32 * channels * (frequency_bands // 4))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(1,2), stride=(1,2), padding=(0,0), output_padding=(0,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(1,2), stride=(1,2), padding=(0,0), output_padding=(0,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.fc1(x)
        x = self.fc2(latent)
        x = x.view(-1,32,5,1)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x, latent

# ===============================
# 5. Training Functions
# ===============================

def train_autoencoder(model, dataloader, config, progress=gr.Progress()):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.to(config.device)
    
    progress_text = ""
    for epoch in range(1, config.eeg_epochs + 1):
        model.train()
        running_loss = 0.0
        
        for data, target in progress.tqdm(dataloader, desc=f"Epoch {epoch}/{config.eeg_epochs}"):
            data = data.to(config.device)
            target = target.to(config.device)

            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        progress_text += f'Epoch {epoch}/{config.eeg_epochs}, Loss: {epoch_loss:.6f}\n'
        
    return model, progress_text

# ===============================
# 6. Hidden Vector Extraction
# ===============================

def extract_hidden_vectors(model, dataloader, config, progress=gr.Progress()):
    model.eval()
    hidden_vectors = []
    with torch.no_grad():
        for data, _ in progress.tqdm(dataloader, desc="Extracting Hidden Vectors"):
            data = data.to(config.device)
            _, latent = model(data)
            hidden_vectors.append(latent.cpu().numpy())
    return np.concatenate(hidden_vectors, axis=0)

# ===============================
# 7. Gradio Interface
# ===============================

def process_eeg(edf_file, epoch_length, batch_size, num_epochs, learning_rate, latent_dim, progress=gr.Progress()):
    # Check if file was uploaded
    if edf_file is None:
        return None, None, "Error: No file uploaded", "Please upload an EDF file"
    
    # Create config with user parameters
    config = Config()
    config.epoch_length = epoch_length
    config.eeg_batch_size = batch_size
    config.eeg_epochs = num_epochs
    config.learning_rate = learning_rate
    config.latent_dim = latent_dim

    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Load EEG data
        raw = mne.io.read_raw_edf(edf_file.name, preload=True, verbose=False)
        progress(0.2, desc="Data loaded")

        # Preprocess EEG data
        epochs_normalized = preprocess_eeg(raw, config)
        progress(0.4, desc="Preprocessing complete")

        # Create dataset and dataloader
        eeg_dataset = EEGDataset(epochs_normalized)
        eeg_dataloader = DataLoader(eeg_dataset, batch_size=config.eeg_batch_size, shuffle=True, num_workers=0)
        
        # Initialize and train model
        channels = epochs_normalized.shape[1]
        frequency_bands_count = epochs_normalized.shape[2]
        model = EEGAutoencoder(channels=channels, frequency_bands=frequency_bands_count, latent_dim=config.latent_dim)
        
        model, training_log = train_autoencoder(model, eeg_dataloader, config, progress)
        progress(0.7, desc="Training complete")

        # Extract hidden vectors
        eeg_dataloader_no_shuffle = DataLoader(eeg_dataset, batch_size=config.eeg_batch_size, shuffle=False, num_workers=0)
        hidden_vectors = extract_hidden_vectors(model, eeg_dataloader_no_shuffle, config, progress)
        progress(0.9, desc="Feature extraction complete")

        # Save outputs
        model_path = os.path.join(temp_dir, 'eeg_autoencoder_model.pth')
        vectors_path = os.path.join(temp_dir, 'hidden_vectors.npy')
        
        torch.save(model.state_dict(), model_path)
        np.save(vectors_path, hidden_vectors)
        
        shape_info = f"Hidden vectors shape: {hidden_vectors.shape}"
        
        return model_path, vectors_path, training_log, shape_info

    except Exception as e:
        error_message = f"Error processing EEG data: {str(e)}"
        return None, None, error_message, error_message

# ===============================
# 8. Gradio App
# ===============================

def create_gradio_interface():
    with gr.Blocks(title="EEG Autoencoder") as app:
        gr.Markdown("# EEG Autoencoder Feature Extractor")
        gr.Markdown("Upload an EDF file and configure parameters to extract features from EEG data.")
        
        with gr.Row():
            with gr.Column():
                edf_file = gr.File(label="Upload EDF File", file_types=[".edf"])
                epoch_length = gr.Number(value=1, label="Epoch Length (seconds)", minimum=0.1)
                batch_size = gr.Number(value=64, label="Batch Size", minimum=1)
                num_epochs = gr.Number(value=50, label="Number of Epochs", minimum=1)
                learning_rate = gr.Number(value=0.001, label="Learning Rate", minimum=0.0001)
                latent_dim = gr.Number(value=64, label="Latent Dimension", minimum=1)
                process_btn = gr.Button("Process EEG Data")
            
            with gr.Column():
                model_output = gr.File(label="Trained Model")
                vectors_output = gr.File(label="Hidden Vectors")
                training_log = gr.Textbox(label="Training Log", lines=10)
                shape_info = gr.Textbox(label="Output Information")

        process_btn.click(
            fn=process_eeg,
            inputs=[edf_file, epoch_length, batch_size, num_epochs, learning_rate, latent_dim],
            outputs=[model_output, vectors_output, training_log, shape_info]
        )

    return app


if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch()