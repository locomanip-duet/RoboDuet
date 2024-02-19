import numpy as np
import torch
import matplotlib.pyplot as plt


device = "cuda:0"

model_path = "./dist/"
hidden_dim = 6
rows = 2
cols = hidden_dim // rows

mean_vector = np.load(model_path+"mean_vec.npy")
cov_mat = np.load(model_path+"cov_mat.npy")
def sample_z():
    sample_np = np.random.multivariate_normal(mean_vector, cov_mat, 409600)
    sample_z = torch.from_numpy(sample_np).float().to(device)
    return(sample_np)
data = sample_z()

fig, axes = plt.subplots(rows, rows, figsize=(15, 10))
for i, ax in enumerate(axes.flatten()):
    ax.hist(data[:, i], bins=80, alpha=0.75)
    ax.set_title(f"dim={i+1} graph")
    ax.set_xlabel('value')
    ax.set_ylabel('frequency')
plt.tight_layout()
plt.show()
plt.savefig(model_path+"distribution.png")

 
