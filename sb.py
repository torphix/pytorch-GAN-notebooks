
import os
import torch
import matplotlib.pyplot as plt

root = '/home/j/Desktop/Programming/DeepLearning/GANs/results/logs/train-run-1/metrics'
metric_files = os.listdir(root)

g_loss, d_loss = [], []
for metric_file in metric_files:
    metrics = torch.load(f'{root}/{metric_file}')
    g_loss.append(metrics['Generator Loss'])
    d_loss.append(metrics['Discriminator Loss'])


plt.plot(g_loss, label='Generator Loss')
plt.plot(d_loss, label='Discriminator Loss')
plt.legend()
plt.show()
