import torch
import matplotlib.pyplot as plt

data = torch.load("../tensors/mahoraga_tensors.pt")
stacked = torch.stack(data)  # shape: (num_samples, 42, 2)

fig, ax = plt.subplots(figsize=(8, 8))

for sample in stacked:
    # First 21 = left hand
    ax.scatter(sample[:21, 0].numpy(), sample[:21, 1].numpy(),
               alpha=0.25, s=10, color="steelblue")
    # Last 21 = right hand
    ax.scatter(sample[21:, 0].numpy(), sample[21:, 1].numpy(),
               alpha=0.25, s=10, color="coral")

# Mean landmarks split the same way
mean_tensor = stacked.mean(dim=0)  # shape: (42, 2)
ax.scatter(mean_tensor[:21, 0].numpy(), mean_tensor[:21, 1].numpy(),
           color="blue", s=40, zorder=5, label="Mean left hand")
ax.scatter(mean_tensor[21:, 0].numpy(), mean_tensor[21:, 1].numpy(),
           color="red", s=40, zorder=5, label="Mean right hand")

ax.set_title("Left hand (blue) vs Right hand (red)")
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.show()