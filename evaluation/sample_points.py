import torch
import numpy as np
from tqdm import tqdm
import os

# Set the ranges for each parameter
l_range = [0.2, 0.8]
p_range = [-torch.pi * 0.5, torch.pi * 0.5]
y_range = [-torch.pi * 0.5, torch.pi * 0.5]
roll_ee_range = [-torch.pi * 0.5, torch.pi * 0.5]
pitch_ee_range = [-torch.pi * 0.5, torch.pi * 0.5]
yaw_ee_range = [-torch.pi * 0.5, torch.pi * 0.5]

# Number of samples to generate in total and per batch
total_samples = 2_000_000
samples_per_batch = 5000
num_batches = total_samples // samples_per_batch

os.makedirs("samples", exist_ok=True)

# Function to scale and shift random numbers to the desired range
def scale_and_shift(tensor, range_min, range_max):
    return tensor * (range_max - range_min) + range_min

# Loop to generate and save samples in batches
for batch_num in tqdm(range(num_batches), desc="生成样本批次"):
    # Generate uniform samples using torch.rand for each parameter
    l_samples = scale_and_shift(torch.rand(samples_per_batch), *l_range)
    p_samples = scale_and_shift(torch.rand(samples_per_batch), *p_range)
    y_samples = scale_and_shift(torch.rand(samples_per_batch), *y_range)
    roll_ee_samples = scale_and_shift(torch.rand(samples_per_batch), *roll_ee_range)
    pitch_ee_samples = scale_and_shift(torch.rand(samples_per_batch), *pitch_ee_range)
    yaw_ee_samples = scale_and_shift(torch.rand(samples_per_batch), *yaw_ee_range)

    # Combine all samples into one tensor
    batch_samples = torch.stack((l_samples, p_samples, y_samples, roll_ee_samples, pitch_ee_samples, yaw_ee_samples), dim=1)
    
    # Convert PyTorch tensor to NumPy array and save as .npy file
    np.save(f'samples/samples_part_{batch_num + 1}.npy', batch_samples.numpy())

print("All samples generated and saved in batches as .npy files.")