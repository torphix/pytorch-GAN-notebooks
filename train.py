import os
import json
import torch
import shutil
import numpy as np
import torchvision
from tqdm import tqdm
import torch.nn as nn
from typing import Dict, Any
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


# Load celebA dataset
def get_dataloader(batch_size, shuffle, n_workers, image_size):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.CelebA(
        root="./data", transform=transform, download=True
    )
    dataset = torch.utils.data.Subset(dataset, [i for i in range(1000)])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers
    )
    return dataloader


class SelfAttentionConv(nn.Module):
    def __init__(self, in_d, downscale_factor=8):
        super().__init__()
        """
        Downscale factor is suggested in the paper to reduce memory consumption
        they use 8, but 4 or 2 can be used with enough gpu memory
        """
        self.downscale_factor = downscale_factor
        self.k_conv = nn.Conv2d(in_d, in_d // self.downscale_factor, 1, 1, 0)
        self.q_conv = nn.Conv2d(in_d, in_d // self.downscale_factor, 1, 1, 0)
        self.v_conv = nn.Conv2d(in_d, in_d, 1, 1, 0)
        # gamma is a learnable parameter (used in original paper)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Conv2d(in_d, in_d, 1, 1, 0)

    def forward(self, x):
        """
        x: (batch_size, in_d, h, w)
        """
        batch_size, in_d, h, w = x.shape
        # Embed input and reshape for matrix multiplication
        k = self.k_conv(x).view(batch_size, -1, h * w)
        q = self.q_conv(x).view(batch_size, -1, h * w)
        v = self.v_conv(x).view(batch_size, -1, h * w)
        # (batch_size, h * w, h * w)
        attn = torch.bmm(k.transpose(-2, -1), q)
        attn = F.softmax(attn, dim=-1)
        # (batch_size, in_d, h * w)
        out = torch.bmm(v, attn.transpose(-2, -1))
        out = out.view(batch_size, in_d, h, w)
        out = self.out_conv(out)
        # Add residual connection and scale by learnable parameter gamma
        out = self.gamma * out + x
        return out, attn


# Generator, DC-GAN with self attention
class Generator(nn.Module):
    def __init__(self, noise_dim=100, target_output_size=128, emb_dim=1024):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, emb_dim, 4),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList()
        n_layers = int(np.log2(target_output_size))
        for _ in range(n_layers - 3):
            self.layers.append(
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.ConvTranspose2d(
                            emb_dim,
                            emb_dim // 2 if emb_dim // 2 >= 64 else emb_dim,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        )
                    ),
                    nn.BatchNorm2d(emb_dim // 2 if emb_dim // 2 >= 64 else emb_dim),
                    nn.ReLU(),
                )
            )
            # Don't shrink emb_dim below too low
            if emb_dim // 2 >= 64:
                emb_dim = emb_dim // 2
        self.layers = nn.Sequential(*self.layers)
        # Self attention sandwhiches the penultimate layer
        self.self_attn_1 = SelfAttentionConv(emb_dim)
        self.self_attn_2 = SelfAttentionConv(emb_dim)
        self.penultimate_layer = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, emb_dim, 3, 1, 1),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
        )
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, noise):
        out = self.in_layer(noise)
        # Main layers
        out = self.layers(out)
        # Attention layers
        out, attn1 = self.self_attn_1(out)
        out = self.penultimate_layer(out)
        out, attn2 = self.self_attn_2(out)
        # Out layers
        out = self.out_layer(out)
        return out, attn1, attn2


# # Discriminator, adopts the PatchGAN architecture ie: output is a receptive field of n x n
class Discriminator(nn.Module):
    def __init__(self, in_d=3, emb_dim=512, n_layers=3):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_d, emb_dim, 4, 2, 1),
            nn.BatchNorm2d(emb_dim),
            nn.LeakyReLU(0.2),
        )
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Conv2d(
                            emb_dim,
                            emb_dim // 2 if emb_dim // 2 >= 64 else emb_dim,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        )
                    ),
                    nn.BatchNorm2d(emb_dim // 2 if emb_dim // 2 >= 64 else emb_dim),
                    nn.LeakyReLU(),
                )
            )
            # Don't shrink emb_dim below too low
            if emb_dim // 2 >= 64:
                emb_dim = emb_dim // 2
        self.layers = nn.Sequential(*self.layers)
        # Self attention sandwhiches the penultimate layer
        self.self_attn_1 = SelfAttentionConv(emb_dim)
        self.self_attn_2 = SelfAttentionConv(emb_dim)
        self.penultimate_layer = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 3, 1, 1),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(),
        )
        self.out_layer = nn.Conv2d(emb_dim, 1, 4)

    def forward(self, noise):
        out = self.in_layer(noise)
        # Main layers
        out = self.layers(out)
        # Attention layers
        out, attn1 = self.self_attn_1(out)
        out = self.penultimate_layer(out)
        out, attn2 = self.self_attn_2(out)
        # Out layers
        out = self.out_layer(out)
        return out, attn1, attn2


class Trainer:
    def __init__(
        self,
        n_ckpt_steps: int,
        n_log_steps: int,
        epochs: int,
        # Data parameters
        batch_size: int,
        # Optimiser parameters
        g_lr: float,
        d_lr: float,
        # Model parameters
        noise_d: int,
        emb_d: int,
        output_size: int,
        flush_prev_logs: bool = True,
    ):
        """
        n_ckpt_steps: Saves a checkpoint every n_ckpt_steps
        n_log_steps: Logs every n_log_steps
        epochs: Number of epochs to train for
        batch_size: Batch size
        lr: Learning rate
        noise_d: Noise dimension
        emb_d: Embedding dimension
        output_size: Output size ie: height and width of the image
        flush_prev_logs: Flushes previous logs and checkpoints
        """

        torch.cuda.empty_cache()
        # Initialize models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(
            noise_dim=noise_d,
            emb_dim=emb_d,
            target_output_size=output_size,
        ).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        # Initialize optimisers
        self.optim_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=g_lr,
        )
        self.optim_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=d_lr,
        )
        self.train_dataloader = get_dataloader(batch_size, True, 4, output_size)
        # Hyperparameters
        self.n_log_steps = n_log_steps
        self.n_ckpt_steps = n_ckpt_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_d = noise_d
        # Init globals & Metrics
        self.global_step = 0
        self.inception = InceptionScore(normalize=True)
        self.fid = FrechetInceptionDistance(normalize=True)
        self.g_losses, self.d_losses = [], []
        # Reset logs
        if flush_prev_logs:
            shutil.rmtree("results/logs", ignore_errors=True)
        # Create log directory
        os.makedirs("results/logs", exist_ok=True)
        self.log_dir = f'results/logs/train-run-{len(os.listdir("results/logs")) + 1}'
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(f"{self.log_dir}/images", exist_ok=True)
        os.makedirs(f"{self.log_dir}/metrics", exist_ok=True)
        os.makedirs(f"{self.log_dir}/checkpoints", exist_ok=True)
        print(f"Logging to {self.log_dir}")

    def to_device(self, *args):
        return [arg.to(self.device) for arg in args]

    def run(self):
        for e in range(self.epochs):
            self.train_iter(e)
        # Save final model
        torch.save(self.generator.state_dict(), f"{self.log_dir}/generator.pt")
        # Save losses
        torch.save(self.g_losses, f"{self.log_dir}/g_losses.pt")
        torch.save(self.d_losses, f"{self.log_dir}/d_losses.pt")
        self.display_metrics(f'{self.log_dir}/metrics')

    def train_iter(self, epoch: int):
        self.generator.train()
        self.discriminator.train()
        # Train on a batch of images
        for i, (real_img, tags) in enumerate(
            tqdm(self.train_dataloader, desc=f"Epoch: {epoch}", leave=False)
        ):
            real_img, tags = real_img.to(self.device), tags.to(self.device)
            # Train the discriminator
            self.optim_D.zero_grad()
            noise = torch.randn((self.batch_size, 100, 1, 1)).to(self.device)
            fake_image, _, _ = self.generator(noise)
            loss_D = self.compute_discriminator_loss(real_img, fake_image)
            loss_D.backward()
            self.optim_D.step()
            # Train the generators
            self.optim_G.zero_grad()
            loss_G = self.compute_generator_loss()
            loss_G.backward()
            self.optim_G.step()
            # Log the losses & Metrics
            if self.global_step % self.n_log_steps == 0:
                self.g_losses.append(loss_G.item())
                self.d_losses.append(loss_D.item())
                self.log_image(real_img, f"{self.global_step}_real_image.png")
                self.log_image(
                    self.generator(noise)[0], f"{self.global_step}_fake_image.png"
                )
                # Compute inception score
                # self.inception.update((self.generator(noise)[0].detach().cpu()))
                # Compute FID
                # self.fid.update(self.generator(noise)[0].detach().cpu(), False)
                # self.fid.update(real_img.cpu(), True)
                # Compute discriminator accuracy
                real_pred_D, _, _ = self.discriminator(real_img)
                fake_pred_D, _, _ = self.discriminator(fake_image)
                real_acc = (torch.sigmoid(real_pred_D) > 0.5).float().mean()
                fake_acc = (torch.sigmoid(fake_pred_D) < 0.5).float().mean()
                # Log the metrics
                self.log_metrics(
                    {
                        "Generator Loss": loss_G.item(),
                        "Discriminator Loss": loss_D.item(),
                        # "Inception Score": self.inception.compute(),
                        # "FID": self.fid.compute(),
                        "Real Accuracy": real_acc.item(),
                        "Fake Accuracy": fake_acc.item(),
                    }
                )
            # Increment the global step
            self.global_step += 1
            # Save a checkpoint
            if self.global_step % self.n_ckpt_steps == 0:
                self.save_checkpoint()

    def compute_generator_loss(self) -> torch.Tensor:
        noise_x = torch.randn(self.batch_size, self.noise_d, 1, 1).to(self.device)
        fake_image, _, _ = self.generator(noise_x)
        fake_pred_D, _, _ = self.discriminator(fake_image)
        # Hinge loss as specified in the paper
        loss_G = -fake_pred_D.mean()
        return loss_G

    def compute_discriminator_loss(
        self,
        real_img: torch.Tensor,
        generated_img: torch.Tensor,
    ) -> torch.Tensor:
        real_pred_D, _, _ = self.discriminator(real_img)
        fake_pred_D, _, _ = self.discriminator(generated_img)
        # Hinge loss as specified in the paper
        loss_D = F.relu(1 - real_pred_D).mean() + F.relu(1 + fake_pred_D).mean()
        return loss_D

    def save_checkpoint(self):
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "optim_G": self.optim_G.state_dict(),
                "optim_D": self.optim_D.state_dict(),
            },
            f"{self.log_dir}/checkpoints/step-{self.global_step}.pt",
        )

    def load_checkpoint(self, path: str):
        state_dict = torch.load(path)
        self.generator.load_state_dict(state_dict["generator"])
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.optim_G.load_state_dict(state_dict["optim_G"])
        self.optim_D.load_state_dict(state_dict["optim_D"])

    def log_image(self, img: torch.Tensor, name: str):
        torchvision.utils.save_image(
            img,
            f"{self.log_dir}/images/{name}",
            normalize=True,
            range=(-1, 1),
        )

    def log_metrics(self, metrics: Dict[str, Any]):
        torch.save(
            metrics, f"{self.log_dir}/metrics/step-{self.global_step}-metrics.pt"
        )

    def display_metrics(self, metrics_dir: str):
        g_loss, d_loss = [], []
        real_acc, fake_acc = [], []
        for file in os.listdir(metrics_dir):
            metrics = torch.load(os.path.join(metrics_dir, file))
            g_loss.append(metrics["Generator Loss"])
            d_loss.append(metrics["Discriminator Loss"])
            real_acc.append(metrics["Real Accuracy"])
            fake_acc.append(metrics["Fake Accuracy"])
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        ax[0, 0].plot(g_loss)
        ax[0, 0].set_title("Generator Loss")
        ax[0, 1].plot(d_loss)
        ax[0, 1].set_title("Discriminator Loss")
        ax[1, 0].plot(real_acc)
        ax[1, 0].set_title("Real Accuracy")
        ax[1, 1].plot(fake_acc)
        ax[1, 1].set_title("Fake Accuracy")
        plt.show()


Trainer(
    n_ckpt_steps=5000,
    n_log_steps=1000,
    epochs=250,
    batch_size=64,
    g_lr=0.0002,
    d_lr=0.0003,
    noise_d=100,
    emb_d=512,
    output_size=64,
    flush_prev_logs=False,
).run()
