import torch
import matplotlib.pyplot as plt
import numpy as np

# Define kappa (you need to define it appropriately)
kappa = 0.07

# Define the smoothing_cdf_start function
smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

# Generate x values for plotting
x = torch.linspace(-2, 2, 400)
y_FL = torch.zeros_like(x)
y_FR = torch.zeros_like(x)
y_RL = torch.zeros_like(x)
y_RR = torch.zeros_like(x)

# Calculate y values for each function
for i, x_val in enumerate(x):
    y_FL[i] = (smoothing_cdf_start(torch.remainder(x_val, 1.0)) * (1 - smoothing_cdf_start(torch.remainder(x_val, 1.0) - 0.5)) +
               smoothing_cdf_start(torch.remainder(x_val, 1.0) - 1) * (1 - smoothing_cdf_start(torch.remainder(x_val, 1.0) - 0.5 - 1)))

    y_FR[i] = (smoothing_cdf_start(torch.remainder(x_val, 1.0)) * (1 - smoothing_cdf_start(torch.remainder(x_val, 1.0) - 0.5)) +
               smoothing_cdf_start(torch.remainder(x_val, 1.0) - 1) * (1 - smoothing_cdf_start(torch.remainder(x_val, 1.0) - 0.5 - 1)))

    y_RL[i] = (smoothing_cdf_start(torch.remainder(x_val, 1.0)) * (1 - smoothing_cdf_start(torch.remainder(x_val, 1.0) - 0.5)) +
               smoothing_cdf_start(torch.remainder(x_val, 1.0) - 1) * (1 - smoothing_cdf_start(torch.remainder(x_val, 1.0) - 0.5 - 1)))

    y_RR[i] = (smoothing_cdf_start(torch.remainder(x_val, 1.0)) * (1 - smoothing_cdf_start(torch.remainder(x_val, 1.0) - 0.5)) +
               smoothing_cdf_start(torch.remainder(x_val, 1.0) - 1) * (1 - smoothing_cdf_start(torch.remainder(x_val, 1.0) - 0.5 - 1)))

# Plot each function
plt.figure(figsize=(10, 6))
plt.plot(x, y_FL, label='FL')
plt.plot(x, y_FR, label='FR')
plt.plot(x, y_RL, label='RL')
plt.plot(x, y_RR, label='RR')
plt.title('Smoothed Functions')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.legend()
plt.grid(True)
plt.show()
