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

from model_gae import GAE

def set_seed(seed):
    random.seed(seed)                         # Python random
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch CPU
    torch.cuda.manual_seed(seed)              # PyTorch GPU
    torch.cuda.manual_seed_all(seed)          # if multi-GPU

def cox_ph_loss(risk_scores, times, events):
    """
    risk_scores: Tensor of shape (N,), output of the model
    times: Tensor of shape (N,), time to event or censoring
    events: Tensor of shape (N,), 1 if event occurred, 0 if censored
    """
    # Sort by time descending
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    # Compute log cumulative sum of exponentials of risk scores
    log_cumsum = torch.logcumsumexp(risk_scores, dim=0)

    # Loss is only for individuals with event == 1
    loss = -torch.sum((risk_scores - log_cumsum) * events)

    return loss / events.sum()  # normalize by number of events


def main(seed, alpha, device):
    set_seed(seed)

    # --- IMPORT DATA ---

    ukb_sasp = pd.read_csv("ukb/ukb_sasp_2.csv")

    ukb_sample = ukb_sasp.iloc[:,7:46]
    ukb_target = ukb_sasp.iloc[:,[3,4, 45]]

    # split sample data by labels
    ukb_sample_train = ukb_sample[ukb_sample['label'] == 0].drop(columns=['label'])
    ukb_sample_val = ukb_sample[ukb_sample['label'] == 1].drop(columns=['label'])
    ukb_sample_test = ukb_sample[ukb_sample['label'] == 2].drop(columns=['label'])

    ukb_target_train = ukb_target[ukb_target['label'] == 0].drop(columns=['label'])
    ukb_target_val = ukb_target[ukb_target['label'] == 1].drop(columns=['label'])
    ukb_target_test = ukb_target[ukb_target['label'] == 2].drop(columns=['label'])

    # Convert to NumPy arrays
    ukb_sample_train = ukb_sample_train.values.astype(np.float32)
    ukb_sample_val = ukb_sample_val.values.astype(np.float32)
    ukb_sample_test = ukb_sample_test.values.astype(np.float32)

    ukb_target_train = ukb_target_train.values.astype(np.float32)
    ukb_target_val = ukb_target_val.values.astype(np.float32)
    ukb_target_test = ukb_target_test.values.astype(np.float32)

    # Convert to Tensor
    ukb_sample_train = torch.tensor(ukb_sample_train).to(device)
    ukb_target_train = torch.tensor(ukb_target_train).to(device)

    ukb_sample_val = torch.tensor(ukb_sample_val).to(device)
    ukb_target_val = torch.tensor(ukb_target_val).to(device)

    # --- MODEL INSTANTIATE ---

    autoencoder = GAE(input_dim=38, latent_dim=6, code_dim=1).to(device)

    optimizer = optim.Adam(
        autoencoder.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    def criterion(recon_x, x, risk_scores, times, events, alpha=0.005):
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        GUD = cox_ph_loss(risk_scores, times, events)
        return alpha * MSE + (1-alpha) * GUD

    # --- TRAIN LOOP ---

    train_losses = []
    test_losses = []

    num_epochs = 500

    for _ in range(num_epochs):

        autoencoder.train()
        scores, recon_x  = autoencoder(ukb_sample_train)

        loss = criterion(recon_x, ukb_sample_train, scores, ukb_target_train[:,1], ukb_target_train[:,0], alpha=alpha)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())

        # --- VALIDATION ---

        autoencoder.eval()

        with torch.no_grad():

            scores, recon_x = autoencoder(ukb_sample_train)

            loss = criterion(recon_x, ukb_sample_train, scores, ukb_target_train[:,1], ukb_target_train[:,0], alpha=alpha)
            test_losses.append(loss.item())

    # --- SAVE TRAINED MODEL ---

    model_filename = f"model_weights/gae_a{alpha}_s{seed}.pth"
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
    plt.savefig(f"loss_plots/loss_a{alpha}_s{seed}.png")

    # --- SAVE SASP INDEX ---

    ukb_sample_all = ukb_sasp.iloc[:,7:45]
    ukb_sample_all = ukb_sample_all.values.astype(np.float32)
    ukb_sample_all = torch.tensor(ukb_sample_all).to(device)

    autoencoder.eval()
    with torch.no_grad():
        sasp_index = autoencoder.predict(ukb_sample_all)
        sasp_index = sasp_index.cpu().numpy()
        df = pd.DataFrame(sasp_index, columns=['sasp_index'])

    ukb_eid = ukb_sasp.iloc[:,0]
    df_combined = pd.concat([ukb_eid, df], axis=1)

    output_filename = f"indexes/index_a{alpha}_s{seed}.csv"
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

    for seed in range(0,5):
        for alpha in [i / 10 for i in range(1, 10)]:
            main(seed = seed, alpha = alpha, device = device)
