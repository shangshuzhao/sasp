import argparse
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from TransformerAutoEncoder import TransformerAE
from protein_to_label import name_to_embed
from rename_medex_protein import rename_medex_columns
from match_dist import match_ukb_dist

def set_seed(seed):
    random.seed(seed)                         # Python random
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch CPU
    torch.cuda.manual_seed(seed)              # PyTorch GPU
    torch.cuda.manual_seed_all(seed)          # if multi-GPU

def prepare_data(train_sample, train_target, valid_sample, valid_target, device):

    # Create a copy of protein name embeddings
    var_names = train_sample.columns.tolist()
    var_label = name_to_embed(var_names)
    var_label = torch.tensor(var_label).to(device)

    # Convert to NumPy arrays
    train_sample = train_sample.values.astype(np.float32)
    valid_sample = valid_sample.values.astype(np.float32)

    train_target = train_target.values.astype(np.float32)
    valid_target = valid_target.values.astype(np.float32)

    # Create TensorDataset
    train_data = TensorDataset(
        torch.tensor(train_sample).to(device),
        torch.tensor(train_target).to(device)
    )

    valid_data = TensorDataset(
        torch.tensor(valid_sample).to(device),
        torch.tensor(valid_target).to(device)
    )

    # Create dataloader
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False)

    return train_loader, valid_loader, var_label

def plot_losses(train_losses, valid_losses, path):

    epochs = range(1, len(train_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses[1:], marker='o', label='Training Error')
    plt.plot(epochs, valid_losses[1:], marker='o', label='Testing Error')

    plt.title('Training and Testing Losses Trend')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)

def criterion(recon_x, x, latent, tl, alpha=0.5):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    GUD = nn.functional.mse_loss(latent, tl, reduction='sum')
    return alpha * MSE + (1-alpha) * GUD

def main(seed, alpha, prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    # --- IMPORT DATA ---
    medex = pd.read_csv("df_medex/MEDEX_Expanded_SASP_ALL_impute_age.csv")

    medex_train_sample = medex.iloc[:1200,4:42]
    medex_valid_sample = medex.iloc[1200:,4:42]
    medex_train_sample = rename_medex_columns(medex_train_sample)
    medex_valid_sample = rename_medex_columns(medex_valid_sample)
    medex_train_sample = match_ukb_dist(medex_train_sample)
    medex_valid_sample = match_ukb_dist(medex_valid_sample)

    medex_train_target = 2 - medex.iloc[:1200, 42] / 50
    medex_valid_target = 2 - medex.iloc[1200:, 42] / 50

    medex_train_loader, medex_valid_loader, proteins_label = prepare_data(medex_train_sample, medex_train_target, medex_valid_sample, medex_valid_target, device)

    # --- NETWORK INSTANTIATE ---

    tgae = TransformerAE().to(device)
    model_path = f"gae_a{str(alpha)[2:]}_s{seed}.pth"
    tgae.load_state_dict(torch.load(model_path))

    # Example learning rates
    high_lr = 1e-3      # For the regressor
    low_lr = 1e-4       # For the rest of the model

    # Define parameter groups
    regressor_params = list(tgae.regressor.parameters())
    other_params = [p for n, p in tgae.named_parameters() if not n.startswith("regressor")]

    # Set up optimizer
    optimizer = torch.optim.Adam([
        {'params': regressor_params, 'lr': high_lr},
        {'params': other_params, 'lr': low_lr}
    ])

    # --- TRAIN LOOP ---

    train_losses = []
    valid_losses = []

    num_epochs = 30

    for i in range(num_epochs):

        tgae.train()
        running_loss = 0.0

        for samples, guides in medex_train_loader:
            latents, recon_x = tgae(samples, proteins_label)
 
            loss = criterion(recon_x, samples, latents, guides, alpha=alpha)

            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        train_losses.append(running_loss / len(medex_train_loader))

        tgae.eval()
        running_loss = 0.0

        with torch.no_grad():
            for samples, guides in medex_valid_loader:
                latents, recon_x = tgae(samples, proteins_label)

                loss = criterion(recon_x, samples, latents, guides, alpha=alpha)
                running_loss += loss.item()

        valid_losses.append(running_loss / len(medex_valid_loader))

    # --- SAVE TRAINED MODEL ---
    m_filename = f"gae_{prefix}_a{str(alpha)[2:]}_s{seed}.pth"
    torch.save(tgae.state_dict(), m_filename)

    # --- PLOT LOSS ---
    p_filename = f"loss_{prefix}_a{str(alpha)[2:]}_s{seed}.png"
    plot_losses(train_losses, valid_losses, p_filename)
    losses = {
        "train losses": train_losses,
        "valid losses": valid_losses
    }
    with open(f"loss_{prefix}_a{str(alpha)[2:]}_s{seed}.json", 'w') as f:
        json.dump(losses, f)

# --- CONFIGURATION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    args = parser.parse_args()

    main(alpha=args.alpha, seed=args.seed, prefix=args.prefix)
