import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mne
from scipy.signal import butter, lfilter
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import logging
import gradio as gr
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, List
import json
from datetime import datetime
import h5py
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import psutil

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ===============================
# 1. Logging Configuration
# ===============================

logging.basicConfig(
    filename='eeg_brain_imaging.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ===============================
# 2. Configuration
# ===============================

class Config:
    def __init__(self):
        self.fs = 100.0
        self.epoch_length = 1
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
        
        # Imaging parameters
        self.grid_size = 32
        self.smoothing_sigma = 1.0
        self.scan_version = "1.0.0"

# ===============================
# 3. Brain Scan Data Structure (Enhanced)
# ===============================

class BrainScan:
    def __init__(self):
        self.source_data = None
        self.confidence_maps = None
        self.hidden_vectors = None  # New attribute for hidden vectors
        self.metadata = {
            'patient_id': None,
            'timestamp': None,
            'scan_parameters': {},
            'model_version': None,
            'grid_size': None
        }
        self.slices = {
            'axial': [],
            'sagittal': [],
            'coronal': []
        }
        
    def save(self, filepath):
        """Save brain scan data, hidden vectors, and metadata to HDF5 format"""
        with h5py.File(filepath, 'w') as f:
            # Save 3D source data
            f.create_dataset('source_data', data=self.source_data)
            f.create_dataset('confidence_maps', data=self.confidence_maps)
            
            # Save hidden vectors if present
            if self.hidden_vectors is not None:
                f.create_dataset('hidden_vectors', data=self.hidden_vectors)
            
            # Save metadata
            meta_grp = f.create_group('metadata')
            for key, value in self.metadata.items():
                if value is not None:
                    if isinstance(value, dict):
                        # Serialize dictionary to JSON string
                        meta_grp.attrs[key] = json.dumps(value)
                    else:
                        meta_grp.attrs[key] = value
            
            # Save slices
            slice_grp = f.create_group('slices')
            for view, slices in self.slices.items():
                if slices:
                    # Ensure all slices have the same shape
                    try:
                        slice_array = np.stack(slices, axis=0)
                        slice_grp.create_dataset(view, data=slice_array)
                    except ValueError as ve:
                        logging.error(f"Error stacking slices for {view}: {ve}")
                        # Optionally, handle variable-sized slices
                        # For now, skip saving inconsistent slices
                        continue

    @classmethod
    def load(cls, filepath):
        """Load brain scan from file"""
        scan = cls()
        with h5py.File(filepath, 'r') as f:
            scan.source_data = f['source_data'][:]
            scan.confidence_maps = f['confidence_maps'][:]
            
            # Load hidden vectors if present
            if 'hidden_vectors' in f:
                scan.hidden_vectors = f['hidden_vectors'][:]
            
            # Load metadata
            meta_grp = f['metadata']
            for key in meta_grp.attrs:
                value = meta_grp.attrs[key]
                # Attempt to deserialize JSON strings back to dicts
                try:
                    deserialized = json.loads(value)
                    scan.metadata[key] = deserialized
                except (json.JSONDecodeError, TypeError):
                    scan.metadata[key] = value
            
            # Load slices
            slice_grp = f['slices']
            for view in slice_grp:
                scan.slices[view] = slice_grp[view][:]
        
        return scan

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
# 5. Signal Processing
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
# 6. Dataset Class
# ===============================

class EEGDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

# ===============================
# 7. Source Mapping (Enhanced)
# ===============================

class BrainMapper:
    def __init__(self, config):
        self.config = config
        self.setup_grid()
        
    def setup_grid(self):
        """Setup 3D grid for brain volume"""
        self.grid_size = self.config.grid_size
        x = np.linspace(-0.1, 0.1, self.grid_size)
        y = np.linspace(-0.1, 0.1, self.grid_size)
        z = np.linspace(-0.1, 0.1, self.grid_size)
        self.grid_points = np.array(np.meshgrid(x, y, z, indexing='ij'))
        self.grid_points_flat = self.grid_points.reshape(3, -1).T
        
        # Create brain mask (ellipsoid)
        x_grid, y_grid, z_grid = self.grid_points
        self.brain_mask = (x_grid**2/0.01 + y_grid**2/0.01 + z_grid**2/0.01) <= 1
        
        # Create major anatomical landmarks
        self.landmarks = {
            'anterior': np.max(y_grid),
            'posterior': np.min(y_grid),
            'superior': np.max(z_grid),
            'inferior': np.min(z_grid),
            'left': np.min(x_grid),
            'right': np.max(x_grid)
        }

    def create_scan_from_hidden_vectors(self, hidden_vectors, model) -> 'BrainScan':
        """Create a new brain scan from multiple hidden vectors"""
        scan = BrainScan()
        
        try:
            # Set number of workers for parallel processing
            n_jobs = min(os.cpu_count() or 1, 4)  # Limit to 4 cores or less
            
            # Process vectors in batches to reduce memory usage
            batch_size = 1000
            num_batches = (len(hidden_vectors) + batch_size - 1) // batch_size
            
            aggregated_sources = np.zeros((self.grid_size, self.grid_size, self.grid_size))
            aggregated_confidences = np.zeros((self.grid_size, self.grid_size, self.grid_size))
            
            for i in tqdm(range(num_batches), desc="Processing vector batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(hidden_vectors))
                batch_vectors = hidden_vectors[start_idx:end_idx]
                
                # Process batch
                for latent_vector in batch_vectors:
                    sources, confidence = self.estimate_sources(latent_vector, model)
                    aggregated_sources += sources
                    aggregated_confidences += confidence
            
            # Average the aggregated data
            if len(hidden_vectors) > 0:
                scan.source_data = aggregated_sources / len(hidden_vectors)
                scan.confidence_maps = aggregated_confidences / len(hidden_vectors)
            else:
                logging.warning("No hidden vectors provided. Scan data will remain zero.")
                scan.source_data = aggregated_sources
                scan.confidence_maps = aggregated_confidences

            scan.hidden_vectors = hidden_vectors
            
            # Set metadata
            scan.metadata.update({
                'timestamp': datetime.now().isoformat(),
                'scan_parameters': {
                    'grid_size': self.grid_size,
                    'smoothing_sigma': self.config.smoothing_sigma,
                    'num_vectors': len(hidden_vectors),
                    'batch_size': batch_size
                },
                'model_version': self.config.scan_version,
                'system_info': {
                    'cpu_count': os.cpu_count(),
                    'memory_available': psutil.virtual_memory().available // (1024 * 1024 * 1024)  # GB
                }
            })
            
            # Generate standard slices
            scan.slices.update(self.generate_standard_slices(scan.source_data))
            
        except Exception as e:
            logging.error(f"Error in create_scan_from_hidden_vectors: {str(e)}")
            raise
            
        return scan
        
    def estimate_sources(self, latent_vector, model):
        """Estimate source locations from latent vector"""
        try:
            with torch.no_grad():
                weights = model.fc2.weight.cpu().numpy()
                projection = weights @ latent_vector
                
                # Interpolate to 3D grid size
                n_sources = self.grid_size**3
                source_estimates = np.interp(
                    np.linspace(0, 1, n_sources),
                    np.linspace(0, 1, len(projection)),
                    np.abs(projection)
                )
                
                # Reshape to 3D grid
                source_estimates = source_estimates.reshape(
                    self.grid_size, self.grid_size, self.grid_size
                )
                
                # Apply brain mask and smoothing
                source_estimates = source_estimates * self.brain_mask
                source_estimates = gaussian_filter(
                    source_estimates, 
                    sigma=self.config.smoothing_sigma
                )
                
                # Normalize
                source_estimates = (source_estimates - source_estimates.min()) 
                source_estimates = source_estimates / (source_estimates.max() + 1e-6)
                
                # Calculate confidence based on spatial gradients
                gradients = np.gradient(source_estimates)
                confidence = gaussian_filter(
                    gradients[0]**2 + gradients[1]**2 + gradients[2]**2, 
                    sigma=self.config.smoothing_sigma
                )
                confidence = confidence / (confidence.max() + 1e-6)
                
            return source_estimates, confidence
            
        except Exception as e:
            logging.error(f"Error in estimate_sources: {str(e)}")
            raise
    
    def generate_standard_slices(self, sources):
        """Generate standard anatomical slices"""
        mid_x = self.grid_size // 2
        mid_y = self.grid_size // 2
        mid_z = self.grid_size // 2
        
        return {
            'axial': [sources[:, :, z] for z in range(self.grid_size)],
            'sagittal': [sources[x, :, :] for x in range(self.grid_size)],
            'coronal': [sources[:, y, :] for y in range(self.grid_size)]
        }
    
# ===============================
# 8. Visualization (Enhanced)
# ===============================

class BrainVisualizer:
    def __init__(self, config):
        self.config = config
        # Set explicit number of cores for joblib
        os.environ['LOKY_MAX_CPU_COUNT'] = str(max(1, (os.cpu_count() or 2) // 2))

    def create_2d_views(self, scan: BrainScan, threshold=0.5) -> go.Figure:
        """Create 2D heatmap visualizations for axial, sagittal, and coronal views."""
        try:
            # Simple version - just show middle slices
            mid_idx = self.config.grid_size // 2
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Axial", "Sagittal", "Coronal"),
                horizontal_spacing=0.05
            )
            
            # Get middle slices from source data directly if slices not available
            if not scan.slices or not scan.slices['axial']:
                axial = scan.source_data[:, :, mid_idx]
                sagittal = scan.source_data[mid_idx, :, :]
                coronal = scan.source_data[:, mid_idx, :]
            else:
                axial = scan.slices['axial'][mid_idx]
                sagittal = scan.slices['sagittal'][mid_idx]
                coronal = scan.slices['coronal'][mid_idx]

            # Add slices
            fig.add_trace(
                go.Heatmap(z=axial, colorscale='RdBu', showscale=True),
                row=1, col=1
            )
            fig.add_trace(
                go.Heatmap(z=sagittal, colorscale='RdBu', showscale=True),
                row=1, col=2
            )
            fig.add_trace(
                go.Heatmap(z=coronal, colorscale='RdBu', showscale=True),
                row=1, col=3
            )
            
            fig.update_layout(height=400, width=1200, title_text="Brain Activity Views")
            return fig
            
        except Exception as e:
            logging.error(f"Error in create_2d_views: {str(e)}")
            return None

    def create_3d_visualization(self, scan: BrainScan, threshold=0.5, marker_size=2) -> go.Figure:
        """Create simplified 3D visualization focusing on active regions.
        
        Args:
            scan (BrainScan): The brain scan data
            threshold (float): Activity threshold for visualization (0-1)
            marker_size (int): Size of the markers in the 3D plot
            
        Returns:
            go.Figure: The 3D visualization figure
        """
        try:
            # Create coordinates
            x, y, z = np.mgrid[0:self.config.grid_size,
                              0:self.config.grid_size,
                              0:self.config.grid_size]
            
            # Mask low activity regions
            mask = scan.source_data > threshold
            
            # Sample points if too many
            total_points = np.sum(mask)
            max_points = 10000
            if total_points > max_points:
                indices = np.where(mask)
                sample_idx = np.random.choice(len(indices[0]), max_points, replace=False)
                x = indices[0][sample_idx]
                y = indices[1][sample_idx]
                z = indices[2][sample_idx]
                values = scan.source_data[x, y, z]
            else:
                values = scan.source_data[mask]
                x, y, z = x[mask], y[mask], z[mask]
            
            # Create 3D scatter plot with customizable marker size
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=marker_size,  # Use the provided marker_size
                    color=values,
                    colorscale='RdBu',
                    opacity=0.8,
                    colorbar=dict(title="Activity")
                ),
                hovertemplate="x: %{x}<br>y: %{y}<br>z: %{z}<br>activity: %{marker.color:.3f}"
            )])
            
            fig.update_layout(
                scene=dict(
                    aspectmode='cube',
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                height=600,
                width=800,
                title=dict(
                    text="3D Brain Activity",
                    x=0.5,
                    y=0.95
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error in create_3d_visualization: {str(e)}")
            return None

    def create_latent_space_plot(self, hidden_vectors, method='tsne') -> go.Figure:
        """Create simplified latent space visualization."""
        try:
            # Use PCA first to reduce to 50 dimensions if vectors are too high-dimensional
            if hidden_vectors.shape[1] > 50:
                pca = PCA(n_components=50)
                hidden_vectors = pca.fit_transform(hidden_vectors)
            
            # Then apply t-SNE for final visualization
            tsne = TSNE(n_components=2, random_state=42)
            embedding = tsne.fit_transform(hidden_vectors)
            
            fig = px.scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                title="Latent Space (t-SNE)",
                labels={'x': 'Component 1', 'y': 'Component 2'}
            )
            
            fig.update_layout(height=400, width=600)
            return fig
            
        except Exception as e:
            logging.error(f"Error in create_latent_space_plot: {str(e)}")
            return None

    def create_hidden_vector_relationship_plot(self, scan: BrainScan) -> go.Figure:
        """Create simplified correlation plot."""
        try:
            if scan.hidden_vectors is None or scan.source_data is None:
                raise ValueError("Missing hidden vectors or source data")
            
            # Calculate mean activity for each time point
            mean_activity = np.mean(scan.source_data.reshape(-1, self.config.grid_size**3), axis=1)
            
            # Ensure vectors match in length
            if len(mean_activity) > len(scan.hidden_vectors):
                mean_activity = mean_activity[:len(scan.hidden_vectors)]
            elif len(mean_activity) < len(scan.hidden_vectors):
                mean_activity = np.pad(mean_activity, (0, len(scan.hidden_vectors) - len(mean_activity)))
            
            # Calculate correlations
            correlations = []
            for i in range(scan.hidden_vectors.shape[1]):
                corr = np.corrcoef(scan.hidden_vectors[:, i], mean_activity)[0, 1]
                correlations.append(corr)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(range(len(correlations))),
                    y=correlations,
                    name='Correlation'
                )
            ])
            
            fig.update_layout(
                title="Hidden Vector-Activity Correlations",
                xaxis_title="Hidden Vector Dimension",
                yaxis_title="Correlation",
                height=400,
                width=600
            )
            
            return fig
            
        except Exception as e:
            logging.error(f"Error in create_hidden_vector_relationship_plot: {str(e)}")
            return None

# ===============================
# 9. Processing Pipeline (Enhanced)
# ===============================

class BrainImagingPipeline:
    def __init__(self, config):
        self.config = config
        self.mapper = BrainMapper(config)
        self.visualizer = BrainVisualizer(config)

    def process_eeg(self, edf_file, model_file, hidden_vectors_file, save_dir=None) -> Tuple['BrainScan', go.Figure, go.Figure]:
        """Process EEG data and create a brain scan."""
        try:
            # Load EEG data
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

            # Load model with weights_only=True for security
            model = EEGAutoencoder(
                channels=5,  # Fixed from working code
                frequency_bands=7,  # Fixed from working code
                latent_dim=64  # Fixed from working code
            )
            
            try:
                # Try loading with weights_only first
                state_dict = torch.load(
                    model_file, 
                    map_location=self.config.device,
                    weights_only=True
                )
            except (RuntimeError, TypeError):
                # Fallback for older saved models
                logging.warning("Loading model with legacy mode. Consider resaving the model.")
                state_dict = torch.load(
                    model_file,
                    map_location=self.config.device
                )

            model.load_state_dict(state_dict)
            model.to(self.config.device)
            model.eval()

            # Load hidden vectors
            hidden_vectors = np.load(hidden_vectors_file)
            if hidden_vectors.ndim != 2 or hidden_vectors.shape[1] != self.config.latent_dim:
                raise ValueError(f"Hidden vectors should be a 2D array with shape (num_samples, {self.config.latent_dim})")

            # Create brain scan
            scan = self.mapper.create_scan_from_hidden_vectors(hidden_vectors, model)

            # Add additional metadata
            scan.metadata.update({
                'edf_file': os.path.basename(edf_file),
                'model_file': os.path.basename(model_file),
                'hidden_vectors_file': os.path.basename(hidden_vectors_file),
                'processing_info': {
                    'cpu_count': int(os.environ.get('LOKY_MAX_CPU_COUNT', 1)),
                    'device': str(self.config.device),
                    'pytorch_version': torch.__version__
                }
            })

            # Save scan if directory provided
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'brain_scan_{timestamp}.h5'
                filepath = os.path.join(save_dir, filename)
                scan.save(filepath)

            # Create visualizations
            fig_2d = self.visualizer.create_2d_views(scan, threshold=0.5)
            fig_3d = self.visualizer.create_3d_visualization(scan, threshold=0.5, marker_size=2)

            return scan, fig_2d, fig_3d

        except Exception as e:
            logging.error(f"Error in processing pipeline: {str(e)}")
            raise
        
    def load_scan(self, scan_file, threshold=0.5) -> Tuple['BrainScan', go.Figure, go.Figure]:
        """
        Load a saved brain scan and generate visualizations.

        Parameters:
            scan_file (str): Path to the saved brain scan file (.h5).
            threshold (float): Activity threshold for visualization.

        Returns:
            scan (BrainScan): The loaded brain scan data.
            fig_2d (go.Figure): The 2D visualization figure.
            fig_3d (go.Figure): The 3D visualization figure.
        """
        try:
            # Load scan
            scan = BrainScan.load(scan_file)

            # Create visualizations
            fig_2d = self.visualizer.create_2d_views(scan, threshold=threshold)
            fig_3d = self.visualizer.create_3d_visualization(scan, threshold=threshold, marker_size=2)

            return scan, fig_2d, fig_3d

        except Exception as e:
            logging.error(f"Error in loading scan: {str(e)}")
            raise

# ===============================
# 10. Visualization (Enhanced)
# ===============================

# (Already defined above)

# ===============================
# 11. Gradio Interface (Enhanced)
# ===============================

def create_interface():
    config = Config()
    pipeline = BrainImagingPipeline(config)
    
    def process_and_visualize(edf_file, model_file, hidden_vectors_file, threshold, save_dir=None):
        try:
            # Validate file uploads
            if edf_file is None or model_file is None or hidden_vectors_file is None:
                raise ValueError("All files (EDF, model, hidden vectors) must be uploaded.")

            # Process EEG data
            scan, fig_2d, fig_3d = pipeline.process_eeg(
                edf_file.name,
                model_file.name,
                hidden_vectors_file.name,
                save_dir=save_dir if save_dir else None
            )
            
            # Create latent space visualization
            fig_latent = pipeline.visualizer.create_latent_space_plot(scan.hidden_vectors, method='tsne')
            
            # Create hidden vector relationship plot
            fig_corr = pipeline.visualizer.create_hidden_vector_relationship_plot(scan)
            
            # Generate statistics
            stats = f"""
            **Scan Information:**
            - **Timestamp:** {scan.metadata['timestamp']}
            - **Grid Size:** {scan.metadata['scan_parameters']['grid_size']}
            - **Maximum Activity:** {np.max(scan.source_data):.3f}
            - **Average Confidence:** {np.mean(scan.confidence_maps):.3f}
            - **Active Regions:** {np.sum(scan.confidence_maps > threshold)}
            - **Number of Hidden Vectors:** {scan.hidden_vectors.shape[0]}
            - **Latent Dimension Size:** {scan.hidden_vectors.shape[1]}
            """
            
            return fig_2d, fig_3d, fig_latent, fig_corr, stats
                
        except Exception as e:
            logging.error(f"Processing Error: {str(e)}")
            return None, None, None, None, f"Error: {str(e)}"

    def load_and_visualize(scan_file, threshold):
        try:
            if scan_file is None:
                raise ValueError("No brain scan file uploaded.")

            # Load scan
            scan = BrainScan.load(scan_file.name)
            
            # Create visualizations
            fig_2d = pipeline.visualizer.create_2d_views(scan, threshold=float(threshold))
            fig_3d = pipeline.visualizer.create_3d_visualization(scan, threshold=float(threshold), marker_size=2)
            
            # Create latent space visualization if hidden vectors are present
            if scan.hidden_vectors is not None:
                fig_latent = pipeline.visualizer.create_latent_space_plot(scan.hidden_vectors, method='tsne')
                fig_corr = pipeline.visualizer.create_hidden_vector_relationship_plot(scan)
            else:
                fig_latent = None
                fig_corr = None
            
            # Generate statistics
            stats = f"""
            **Loaded Scan Information:**
            - **Original Timestamp:** {scan.metadata['timestamp']}
            - **Grid Size:** {scan.metadata['scan_parameters']['grid_size']}
            - **Maximum Activity:** {np.max(scan.source_data):.3f}
            - **Average Confidence:** {np.mean(scan.confidence_maps):.3f}
            - **Active Regions:** {np.sum(scan.confidence_maps > threshold)}
            - **Number of Hidden Vectors:** {scan.hidden_vectors.shape[0] if scan.hidden_vectors is not None else 'N/A'}
            - **Latent Dimension Size:** {scan.hidden_vectors.shape[1] if scan.hidden_vectors is not None else 'N/A'}
            """
            
            return fig_2d, fig_3d, fig_latent, fig_corr, stats
                
        except Exception as e:
            logging.error(f"Loading Error: {str(e)}")
            return None, None, None, None, f"Error: {str(e)}"
    
    # Create interface
    with gr.Blocks(title="EEG Brain Imaging System") as app:
        gr.Markdown("""
        # EEG Brain Imaging System
        Upload EEG data and a trained model to generate and explore comprehensive 2D and 3D brain activity visualizations, along with insights into the latent space of the model.
        """)
        
        with gr.Tabs():
            # New Scan Tab
            with gr.TabItem("New Scan"):
                with gr.Row():
                    with gr.Column():
                        edf_input = gr.File(label="EEG Data (.edf)", file_types=['.edf'])
                        model_input = gr.File(label="Trained Model (.pth)", file_types=['.pth'])
                        hidden_vectors_input = gr.File(label="Hidden Vectors (.npy)", file_types=['.npy'])
                        threshold = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.01,
                            label="Activity Threshold"
                        )
                        save_dir = gr.Textbox(
                            label="Save Directory (optional)",
                            placeholder="Leave empty to skip saving"
                        )
                        process_button = gr.Button("Generate Brain Scan")
                    
                    with gr.Column():
                        gr.Markdown("## 2D Brain Activity Views")
                        plot_2d = gr.Plot(label="2D Visualization")
                        gr.Markdown("## 3D Brain Activity Visualization")
                        plot_3d = gr.Plot(label="3D Visualization")
                        gr.Markdown("## Latent Space Visualization")
                        plot_latent = gr.Plot(label="Latent Space (t-SNE)")
                        gr.Markdown("## Latent Dimension Correlation")
                        plot_corr = gr.Plot(label="Latent Dimension Correlation")
                        stats_output = gr.Markdown(label="Scan Statistics")
                
                process_button.click(
                    fn=process_and_visualize,
                    inputs=[edf_input, model_input, hidden_vectors_input, threshold, save_dir],
                    outputs=[plot_2d, plot_3d, plot_latent, plot_corr, stats_output]
                )
            
            # Load Scan Tab
            with gr.TabItem("Load Scan"):
                with gr.Row():
                    with gr.Column():
                        scan_input = gr.File(label="Brain Scan (.h5)", file_types=['.h5'])
                        threshold_input = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.01,
                            label="Activity Threshold"
                        )
                        load_button = gr.Button("Load Brain Scan")
                    
                    with gr.Column():
                        gr.Markdown("## 2D Brain Activity Views")
                        load_plot_2d = gr.Plot(label="2D Visualization")
                        gr.Markdown("## 3D Brain Activity Visualization")
                        load_plot_3d = gr.Plot(label="3D Visualization")
                        gr.Markdown("## Latent Space Visualization")
                        load_plot_latent = gr.Plot(label="Latent Space (t-SNE)")
                        gr.Markdown("## Latent Dimension Correlation")
                        load_plot_corr = gr.Plot(label="Latent Dimension Correlation")
                        load_stats_output = gr.Markdown(label="Scan Statistics")
                
                load_button.click(
                    fn=load_and_visualize,
                    inputs=[scan_input, threshold_input],
                    outputs=[load_plot_2d, load_plot_3d, load_plot_latent, load_plot_corr, load_stats_output]
                )
        
        gr.Markdown("""
        ### Instructions
        1. **New Scan:**
           - Upload EEG data file (`.edf`), a trained model file (`.pth`), and the corresponding hidden vectors file (`.npy`).
           - Adjust the activity threshold to highlight regions of interest.
           - Optionally specify a save directory to store the generated scan.
           - Click **Generate Brain Scan** to visualize the results in 2D, 3D, and explore the latent space.
    
        2. **Load Scan:**
           - Upload a previously saved brain scan file (`.h5`).
           - Adjust the activity threshold as needed.
           - Click **Load Brain Scan** to explore the scan in 2D, 3D, and view latent space relationships.
        """)

    return app

# ===============================
# 12. Main Execution
# ===============================

if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)