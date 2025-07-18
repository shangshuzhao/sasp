import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from protein_to_label import name_to_embed
from TransformerAutoEncoder import TransformerAE

def set_seed(seed):
    random.seed(seed)                         # Python random
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch CPU
    torch.cuda.manual_seed(seed)              # PyTorch GPU
    torch.cuda.manual_seed_all(seed)          # if multi-GPU

def prepare_data(ukb_sasp_train, ukb_sasp_val, device):

    # Extract proteins data and guidance data
    ukb_sample_train = ukb_sasp_train.iloc[:,7:]
    ukb_sample_val = ukb_sasp_val.iloc[:,7:]

    ukb_target_train = ukb_sasp_train.iloc[:,5]
    ukb_target_val = ukb_sasp_val.iloc[:,5]

    # Embed variables names
    var_names = ukb_sample_train.columns.tolist()
    var_label = name_to_embed(var_names)
    var_label = torch.tensor(var_label).to(device)

    # Convert to NumPy arrays
    ukb_sample_train = ukb_sample_train.values.astype(np.float32)
    ukb_sample_val = ukb_sample_val.values.astype(np.float32)

    ukb_target_train = ukb_target_train.values.astype(np.float32)
    ukb_target_val = ukb_target_val.values.astype(np.float32)

    # Create TensorDataset
    ukb_train = TensorDataset(
        torch.tensor(ukb_sample_train).to(device),
        torch.tensor(ukb_target_train).to(device)
    )

    ukb_val = TensorDataset(
        torch.tensor(ukb_sample_val).to(device),
        torch.tensor(ukb_target_val).to(device)
    )

    # Create dataloader
    ukb_train_loader = DataLoader(ukb_train, batch_size=128, shuffle=True)
    ukb_val_loader = DataLoader(ukb_val, batch_size=128, shuffle=False)

    return ukb_train_loader, ukb_val_loader, var_label

def criterion(recon_x, x, latent, tl, alpha=0.5):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    GUD = nn.functional.mse_loss(latent, tl, reduction='sum')
    return alpha * MSE + (1-alpha) * GUD

def plot_losses(train_losses, test_losses, alpha, seed):

    epochs = range(1, len(train_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses[1:], marker='o', label='Training Error')
    plt.plot(epochs, test_losses[1:], marker='o', label='Testing Error')

    plt.title('Training and Testing Losses Trend')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_a{str(alpha)[2:]}_s{seed}.png")

def save_sasp_index(tgae, ukb_sasp_all, alpha, seed, device):

    ukb_sample_all = ukb_sasp_all.iloc[:,7:45]
    ukb_sample_all = ukb_sample_all.values.astype(np.float32)
    ukb_sample_all = torch.tensor(ukb_sample_all).to(device)

    tgae.eval()
    with torch.no_grad():
        ae_sasp_index = tgae.predict(ukb_sample_all)

    sasp_index = ae_sasp_index.cpu().numpy()
    df = pd.DataFrame(sasp_index, columns=['sasp_index'])

    ukb_eid = ukb_sasp_all.iloc[:,0]
    df_combined = pd.concat([ukb_eid, df], axis=1)

    output_filename = f"index_a{str(alpha)[2:]}_s{seed}.csv"
    df_combined.to_csv(output_filename, index=False)

def main(seed, alpha):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    # --- IMPORT DATA ---

    ukb_sasp_train = pd.read_csv("df_ukb/ukb_sasp_train.csv")
    ukb_sasp_val = pd.read_csv("df_ukb/ukb_sasp_val.csv")
    ukb_train_loader, ukb_val_loader, proteins_label = prepare_data(ukb_sasp_train, ukb_sasp_val, device)

    # --- NETWORK INSTANTIATE ---

    tgae = TransformerAE().to(device)

    optimizer = optim.Adam(
        tgae.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    # --- TRAIN LOOP ---

    train_losses = []
    test_losses = []

    num_epochs = 250

    for _ in range(num_epochs):

        tgae.train()
        running_loss = 0.0

        for samples, guides in ukb_train_loader:
            latents, recon_x = tgae(samples, proteins_label)

            loss = criterion(recon_x, samples, latents, guides, alpha=alpha)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        train_losses.append(running_loss / len(ukb_train_loader))

        tgae.eval()
        running_loss = 0.0

        with torch.no_grad():
            for samples, guides in ukb_val_loader:
                latents, recon_x = tgae(samples, proteins_label)

                running_loss += criterion(recon_x, samples, latents, guides, alpha=alpha).item()

        test_losses.append(running_loss / len(ukb_val_loader))

    # --- SAVE TRAINED MODEL ---
    model_filename = f"gae_a{str(alpha)[2:]}_s{seed}.pth"
    torch.save(tgae.state_dict(), model_filename)

    # --- PLOT LOSS ---
    plot_losses(train_losses, test_losses, alpha, seed)

# --- CONFIGURATION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    args = parser.parse_args()

    main(seed=args.seed, alpha=args.alpha)
