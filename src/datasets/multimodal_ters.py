import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class MultimodalTERSDataset(Dataset):
    """
    Multimodal TERS Dataset that returns:
    - image: spectral channels (C, H, W)
    - frequencies: padded frequency vector (max_freqs,)
    - freq_mask: mask indicating valid frequencies (max_freqs,)
    - target: segmentation target (4, H, W)
    """
    def __init__(self, hdf5_path, num_channels=400, max_freqs=60, t_image=None, train_aug=False, freq_encoding='binning'):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.num_channels = num_channels
        self.max_freqs = max_freqs
        self.t_image = t_image
        self.train_aug = train_aug
        self.hf = h5py.File(hdf5_path, 'r')
        channels_key = f'channels_{num_channels}'
        self.channels = self.hf[channels_key]
        self.targets = self.hf['targets']
        self.length = self.channels.shape[0]
        self.indices = list(self.hf['frequencies'].keys())
        print(f"Loaded {self.length} samples from {hdf5_path}")
        self.freq_encoding = freq_encoding  # 'binning' or 'normalize'
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        image = self.channels[idx]
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        target = self.targets[idx]
        target = torch.from_numpy(target).float()
        str_idx = self.indices[idx]
        frequencies = np.array(self.hf['frequencies'][str_idx])


        # Frequency preprocessing
        n_freqs = len(frequencies)
        if self.freq_encoding == 'normalize':
            # Normalization and padding approach
            frequencies_norm = frequencies / 4000.0
            freq_padded = np.zeros(self.max_freqs, dtype=np.float32)
            freq_mask = np.zeros(self.max_freqs, dtype=np.float32)
            n_freqs = min(len(frequencies), self.max_freqs)
            freq_padded[:n_freqs] = frequencies_norm[:n_freqs]
            freq_mask[:n_freqs] = 1.0
            freq_padded = torch.from_numpy(freq_padded)
            freq_mask = torch.from_numpy(freq_mask)
            freq_feat = freq_padded
        else:
            # Default: single 400-dim multi-hot binning
            num_bins = 400
            bin_width = 4000.0 / num_bins
            freq_binned = np.zeros(num_bins, dtype=np.float32)
            for freq in frequencies:
                bin_idx = int(freq // bin_width)
                bin_idx = min(bin_idx, num_bins - 1)
                freq_binned[bin_idx] = 1.0
            freq_feat = torch.from_numpy(freq_binned)
            freq_mask = None  # Not needed for single vector


        
        if self.t_image is not None:
            image = self.t_image(image)
        out = {
            'image': image,
            'frequencies': freq_feat,
            'target': target,
            'n_freqs': n_freqs
        }
        if self.freq_encoding == 'normalize':
            out['freq_mask'] = freq_mask
        return out
    def close(self):
        self.hf.close()
