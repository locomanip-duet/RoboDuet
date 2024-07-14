import torch
from tqdm import tqdm

# Set the ranges for each parameter
l_range = [0.2, 0.8]
p_range = [-torch.pi * 0.5, torch.pi * 0.5]
y_range = [-torch.pi * 0.5, torch.pi * 0.5]
roll_ee_range = [-torch.pi * 0.5, torch.pi * 0.5]
pitch_ee_range = [-torch.pi * 0.5, torch.pi * 0.5]
yaw_ee_range = [-torch.pi * 0.5, torch.pi * 0.5]

# Number of samples to generate in total and per batch
total_samples = 2_000_000
num_batches = 200
samples_per_batch = total_samples // num_batches

# Function to scale and shift random numbers to the desired range
def scale_and_shift(tensor, range_min, range_max):
    return tensor * (range_max - range_min) + range_min

# Loop to generate and save samples in batches
for batch_num in range(num_batches):
    # Generate uniform samples using torch.rand for each parameter
    l_samples = scale_and_shift(torch.rand(samples_per_batch), *l_range)
    p_samples = scale_and_shift(torch.rand(samples_per_batch), *p_range)
    y_samples = scale_and_shift(torch.rand(samples_per_batch), *y_range)
    roll_ee_samples = scale_and_shift(torch.rand(samples_per_batch), *roll_ee_range)
    pitch_ee_samples = scale_and_shift(torch.rand(samples_per_batch), -pitch_ee_range[0], pitch_ee_range[1])
    yaw_ee_samples = scale_and_shift(torch.rand(samples_per_batch), -yaw_ee_range[0], yaw_ee_range[1])

    # Combine all samples into one tensor
    batch_samples = torch.stack((l_samples, p_samples, y_samples, roll_ee_samples, pitch_ee_samples, yaw_ee_samples), dim=1)
    
    # Save the batch samples to a file
    torch.save(batch_samples, f'samples/samples_part_{batch_num + 1}.pt')

print("All samples generated and saved in batches.")