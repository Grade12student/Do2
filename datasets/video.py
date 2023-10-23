import os
import numpy as np
import logging
import torch
import cv2  # Add this import statement to resolve the NameError

from torch.utils.data import Dataset

class SingleVideoDataset(Dataset):
    def __init__(self, frames, transforms=None, subset_pct=0.2, num_sparse_frames=None):
        super(SingleVideoDataset, self).__init__()

        self.frames = frames if frames is not None else []  # Ensure self.frames is a list
        self.transforms = transforms

        # Sparse frame sampling configurations
        self.subset_pct = subset_pct
        self.num_sparse_frames = num_sparse_frames

    def __len__(self):
        return len(self.frames)  # Return the length of self.frames

    def __getitem__(self, idx):
        if self.num_sparse_frames is not None:
            num_sparse_frames = self.num_sparse_frames
        else:
            num_sparse_frames = int(len(self.frames) * self.subset_pct)

        # Sample sparse frames uniformly from the full set of frames
        indices = torch.randperm(len(self.frames))[:num_sparse_frames]
        sparse_frames = [torch.from_numpy(self.frames[i]) for i in indices]  # Convert NumPy array to PyTorch tensor

        input_frames = torch.stack(sparse_frames).float() / 255  # Preprocess frames as needed

        # Get the original full frame corresponding to the input frame
        target_frame = torch.from_numpy(self.frames[idx])  # Convert NumPy array to PyTorch tensor

        # Apply transformations if required
        if self.transforms:
            input_frames = self.transforms(input_frames)
            target_frame = self.transforms(target_frame)

        return input_frames, target_frame


# Load frames from the "frames" directory
frames_directory = "framess"  # Update this with the path to your frames directory

frames = []
for filename in os.listdir(frames_directory):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        image_path = os.path.join(frames_directory, filename)
        frame = cv2.imread(image_path)
        frames.append(frame)

dataset = SingleVideoDataset(frames)
