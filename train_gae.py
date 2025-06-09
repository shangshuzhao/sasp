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

from GAE_model import GAE

def set_seed(seed):
    random.seed(seed)                         # Python random
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch CPU
    torch.cuda.manual_seed(seed)              # PyTorch GPU
    torch.cuda.manual_seed_all(seed)          # if multi-GPU

def main(seed, alpha, device):
    set_seed(seed)

    # --- IMPORT DATA ---

    ukb_sasp = pd.read_csv("ukb/ukb_sasp_2.csv")

    ukb_sample = ukb_sasp.iloc[:,7:46]
    ukb_target = ukb_sasp.iloc[:,[5, 45]]

    # # Standardize telomere length data
    # ukb_target = ukb_target.assign(
    #     tl_std  = (ukb_target.iloc[:,0] - ukb_target.iloc[:,0].mean()) / ukb_target.iloc[:,0].std()
    # )
    # ukb_target = ukb_target.iloc[:,[1,2]]

    # split sample data by labels
    ukb_sample_train = ukb_sample[ukb_sample['label'] == 0].drop(columns=['label'])
    ukb_sample_val = ukb_sample[ukb_sample['label'] == 1].drop(columns=['label'])
    ukb_sample_test = ukb_sample[ukb_sample['label'] == 2].drop(columns=['label'])

    # split target data by labels
    ukb_target_train = ukb_target[ukb_target['label'] == 0].drop(columns=['label'])
    ukb_target_val = ukb_target[ukb_target['label'] == 1].drop(columns=['label'])
    ukb_target_test = ukb_target[ukb_target['label'] == 2].drop(columns=['label'])

    # Convert to NumPy arrays
    ukb_sample_train = ukb_sample_train.values.astype(np.float32)
    ukb_sample_val = ukb_sample_val.values.astype(np.float32)
    ukb_sample_test = ukb_sample_test.values.astype(np.float32)

    # Convert to NumPy arrays
    ukb_target_train = ukb_target_train.values.astype(np.float32)
    ukb_target_val = ukb_target_val.values.astype(np.float32)
    ukb_target_test = ukb_target_test.values.astype(np.float32)

    ukb_train = TensorDataset(torch.tensor(ukb_sample_train).to(device),
                            torch.tensor(ukb_target_train).to(device))

    ukb_val = TensorDataset(torch.tensor(ukb_sample_val).to(device),
                            torch.tensor(ukb_target_val).to(device))

    ukb_test = TensorDataset(torch.tensor(ukb_sample_test).to(device),
                            torch.tensor(ukb_target_test).to(device))

    # Load the data to dataloader
    ukb_train_loader = DataLoader(ukb_train, batch_size=128, shuffle=True)
    ukb_val_loader = DataLoader(ukb_val, batch_size=128, shuffle=False)
    ukb_test_loader = DataLoader(ukb_test, batch_size=128, shuffle=False)

    # --- NETWORK INSTANTIATE ---

    autoencoder = GAE(input_dim=38, latent_dim=6, code_dim=1).to(device)

    optimizer = optim.Adam(
        autoencoder.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    def criterion(recon_x, x, latent, tl, alpha=0.5):
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        GUD = nn.functional.mse_loss(latent, tl, reduction='sum')
        return alpha * MSE + (1-alpha) * GUD

    # --- TRAIN LOOP ---

    train_losses = []
    test_losses = []

    num_epochs = 2000

    for _ in range(num_epochs):

        autoencoder.train()
        running_loss = 0.0

        for samples, targets in ukb_train_loader:
            latents, recon_x  = autoencoder(samples)

            loss = criterion(recon_x, samples, latents, targets, alpha=alpha)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(ukb_train_loader)
        train_losses.append(avg_train_loss)

        autoencoder.eval()
        running_loss = 0.0

        with torch.no_grad():
            for samples, targets in ukb_val_loader:
                latents, recon_x = autoencoder(samples)

                running_loss += criterion(recon_x, samples, latents, targets, alpha=alpha).item()

        avg_test_loss = running_loss / len(ukb_val_loader)
        test_losses.append(avg_test_loss)

    # --- SAVE TRAINED MODEL ---

    model_filename = f"gae_s{seed}_a{alpha}.pth"
    torch.save(autoencoder.state_dict(), model_filename)

    # --- PLOT LOSS ---

    epochs = range(1, len(train_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses[1:], marker='o', label='Training Error')
    plt.plot(epochs, test_losses[1:], marker='o', label='Testing Error')

    plt.title('Training and Testing Losses Trend')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_s{seed}_a{alpha}.png")

    # --- SAVE SASP INDEX ---

    ukb_sample_all = ukb_sasp.iloc[:,7:45]
    ukb_sample_all = ukb_sample_all.values.astype(np.float32)
    ukb_sample_all = torch.tensor(ukb_sample_all).to(device)

    autoencoder.eval()
    with torch.no_grad():
        ae_sasp_index = autoencoder.predict(ukb_sample_all)

    sasp_index = ae_sasp_index.cpu().numpy()
    df = pd.DataFrame(sasp_index, columns=['sasp_index'])

    ukb_eid = ukb_sasp.iloc[:,0]
    df_combined = pd.concat([ukb_eid, df], axis=1)

    output_filename = f"index_s{seed}_a{alpha}.csv"
    df_combined.to_csv(output_filename, index=False)

# --- CONFIGURATION ---

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     '--seed',
    #     type=int,
    #     required=True
    # )

    # args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for alpha in [i / 20 for i in range(1, 5)]:
        main(seed = 42, alpha = alpha, device = device)
