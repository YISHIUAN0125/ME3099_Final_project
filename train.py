import os
import itertools
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from itertools import cycle
from model import Generator, Discriminator
from transform import transform_train
from utils.dataloader import FlatFolderDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets & Dataloaders
dataA = FlatFolderDataset("cycleGAN_dataset/processed_unpair_images", transform=transform_train)
dataB = FlatFolderDataset("cycleGAN_dataset/processed_xray_images", transform=transform_train)
loaderA = DataLoader(dataA, batch_size=10, shuffle=True, drop_last=True, num_workers=4)
loaderB = DataLoader(dataB, batch_size=10, shuffle=True, drop_last=True,num_workers=4)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("images", exist_ok=True)

G_AB = Generator(3, 3, 6).to(device)
G_BA = Generator(3, 3, 6).to(device)
D_A = Discriminator(3).to(device)
D_B = Discriminator(3).to(device)

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.0004, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

losses = {"G": [], "D_A": [], "D_B": []}

# AMP scaler
scaler_G = torch.GradScaler("cuda")
scaler_D_A = torch.GradScaler("cuda")
scaler_D_B = torch.GradScaler("cuda")

if __name__ == "__main__":
    print("Training CycleGAN with AMP...")
    EPOCHS = 200
    for epoch in range(1, EPOCHS + 1):
        total_loss_G = 0.0
        total_loss_D_A = 0.0
        total_loss_D_B = 0.0

        progress = tqdm(zip(cycle(loaderA), loaderB), total=len(loaderB), desc=f"Epoch {epoch}")

        for (real_A, _), (real_B, _) in progress:
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            valid = torch.ones((real_A.size(0), *D_A(real_A).shape[1:]), device=device)
            fake = torch.zeros_like(valid)

            # === Generators ===
            optimizer_G.zero_grad()
            with torch.autocast("cuda"):
                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)
                recov_A = G_BA(fake_B)
                recov_B = G_AB(fake_A)

                loss_id_A = criterion_identity(G_BA(real_A), real_A)
                loss_id_B = criterion_identity(G_AB(real_B), real_B)
                loss_identity = (loss_id_A + loss_id_B) * 5.0

                loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
                loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
                loss_GAN = loss_GAN_AB + loss_GAN_BA

                loss_cycle_A = criterion_cycle(recov_A, real_A)
                loss_cycle_B = criterion_cycle(recov_B, real_B)
                loss_cycle = (loss_cycle_A + loss_cycle_B) * 10.0

                loss_G = loss_GAN + loss_cycle + loss_identity

            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            # === Discriminator A ===
            optimizer_D_A.zero_grad()
            with torch.autocast("cuda"):
                loss_real_A = criterion_GAN(D_A(real_A), valid)
                loss_fake_A = criterion_GAN(D_A(fake_A.detach()), fake)
                loss_D_A = (loss_real_A + loss_fake_A) * 0.5

            scaler_D_A.scale(loss_D_A).backward()
            scaler_D_A.step(optimizer_D_A)
            scaler_D_A.update()

            # === Discriminator B ===
            optimizer_D_B.zero_grad()
            with torch.autocast("cuda"):
                loss_real_B = criterion_GAN(D_B(real_B), valid)
                loss_fake_B = criterion_GAN(D_B(fake_B.detach()), fake)
                loss_D_B = (loss_real_B + loss_fake_B) * 0.5

            scaler_D_B.scale(loss_D_B).backward()
            scaler_D_B.step(optimizer_D_B)
            scaler_D_B.update()

            total_loss_G += loss_G.item()
            total_loss_D_A += loss_D_A.item()
            total_loss_D_B += loss_D_B.item()

        n_batches = len(loaderA)
        avg_G = total_loss_G / n_batches
        avg_D_A = total_loss_D_A / n_batches
        avg_D_B = total_loss_D_B / n_batches
        losses["G"].append(avg_G)
        losses["D_A"].append(avg_D_A)
        losses["D_B"].append(avg_D_B)

        if epoch % 50 == 0:
            imgs = torch.cat([
                real_A[:2], fake_B[:2], recov_A[:2],
                real_B[:2], fake_A[:2], recov_B[:2]
            ])
            imgs = make_grid(imgs, nrow=2, normalize=True)
            save_image(imgs, f"images/epoch_{epoch}.png")

            torch.save(G_AB.state_dict(), f"checkpoints/G_AB_epoch{epoch}.pth")
            torch.save(G_BA.state_dict(), f"checkpoints/G_BA_epoch{epoch}.pth")
            torch.save(D_A.state_dict(), f"checkpoints/D_A_epoch{epoch}.pth")
            torch.save(D_B.state_dict(), f"checkpoints/D_B_epoch{epoch}.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(losses["G"], label="Generator")
    plt.plot(losses["D_A"], label="Discriminator A")
    plt.plot(losses["D_B"], label="Discriminator B")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("CycleGAN Training Loss (AMP)")
    plt.savefig("checkpoints/loss_curve_amp.png")
    plt.close()
