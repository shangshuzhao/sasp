import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def embed_protein(name_list: list) -> list:
    ''' Convert protein names to embedding indices '''
    # Predefined list of 38 expected proteins
    name_dict = {
        'ANG': 0,
        'CCL13': 1,
        'CCL2': 2,
        'CCL20': 3,
        'CCL3': 4,
        'CCL4': 5,
        'CHI3L1': 6,
        'CSF2': 7,
        'CXCL1': 8,
        'CXCL10': 9,
        'CXCL8': 10,
        'CXCL9': 11,
        'FGA': 12,
        'FGF2': 13,
        'FSTL3': 14,
        'GDF15': 15,
        'HGF': 16,
        'ICAM1': 17,
        'IGFBP2': 18,
        'IGFBP6': 19,
        'IL1B': 20,
        'IL4': 21,
        'IL5': 22,
        'IL6': 23,
        'IL6ST': 24,
        'LEP': 25,
        'MIF': 26,
        'MMP12': 27,
        'PGF': 28,
        'PLAUR': 29,
        'SERPINE1': 30,
        'TF': 31,
        'TIMP1': 32,
        'TNFRSF11B': 33,
        'TNFRSF1A': 34,
        'TNFRSF1B': 35,
        'TNFSF10': 36,
        'VEGFA': 37
    }
    name_embed = [name_dict[name] for name in name_list]
    return name_embed

def embed_to_ukb_protein(name_list: list) -> list:
    ''' Convert embedding indices to protein names'''
    embed_to_name = {
        0: 'ANG',
        1: 'CCL13',
        2: 'CCL2',
        3: 'CCL20',
        4: 'CCL3',
        5: 'CCL4',
        6: 'CHI3L1',
        7: 'CSF2',
        8: 'CXCL1',
        9: 'CXCL10',
        10: 'CXCL8',
        11: 'CXCL9',
        12: 'FGA',
        13: 'FGF2',
        14: 'FSTL3',
        15: 'GDF15',
        16: 'HGF',
        17: 'ICAM1',
        18: 'IGFBP2',
        19: 'IGFBP6',
        20: 'IL1B',
        21: 'IL4',
        22: 'IL5',
        23: 'IL6',
        24: 'IL6ST',
        25: 'LEP',
        26: 'MIF',
        27: 'MMP12',
        28: 'PGF',
        29: 'PLAUR',
        30: 'SERPINE1',
        31: 'TF',
        32: 'TIMP1',
        33: 'TNFRSF11B',
        34: 'TNFRSF1A',
        35: 'TNFRSF1B',
        36: 'TNFSF10',
        37: 'VEGFA'
    }
    name_embed = [embed_to_name[name] for name in name_list]
    return name_embed

def set_seed(seed):
    ''' Set random seed for reproducibility '''
    random.seed(seed)                         # Python random
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch CPU
    torch.cuda.manual_seed(seed)              # PyTorch GPU
    torch.cuda.manual_seed_all(seed)          # if multi-GPU

def prepare_data(train_sample, train_target, valid_sample, valid_target, device):
    ''' Prepare dataloader and protein name embeddings '''
    # Create a copy of protein name embeddings
    var_names = train_sample.columns.tolist()
    var_label = embed_protein(var_names)
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
    ''' Plot training and validation losses '''
    assert len(train_losses) == len(valid_losses), "Train and valid losses must have the same length."
    if len(train_losses) < 2:
        raise ValueError("Need at least 2 epochs of losses to plot.")
    
    epochs = range(len(train_losses)-1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses[1:], marker='o', label='Training Error')
    plt.plot(epochs, valid_losses[1:], marker='o', label='Testing Error')

    plt.title('Training and Testing Losses Trend')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def criterion(recon_x, x, latent, tl, alpha=0.06):
    ''' Combined loss function: alpha * MSE + (1-alpha) * GUD '''
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    GUD = nn.functional.mse_loss(latent, tl, reduction='sum')
    return alpha * MSE + (1-alpha) * GUD

def training_sasp(num_epochs, tgae, optimizer, train_loader, valid_loader, embeddings, alpha=0.005):
    ''' Training loop for SASP Transformer GAE '''

    train_losses = []
    valid_losses = []

    for _ in range(num_epochs):

        # ---- TRAINING ----

        tgae.train()
        running_loss = 0.0

        for samples, guides in train_loader:
            latents, recon_x = tgae(samples, embeddings)

            loss = criterion(recon_x, samples, latents, guides, alpha=alpha)
            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        train_losses.append(running_loss / len(train_loader))

        # ---- VALIDATION ----

        tgae.eval()
        running_loss = 0.0

        with torch.no_grad():
            for samples, guides in valid_loader:
                latents, recon_x = tgae(samples, embeddings)

                loss = criterion(recon_x, samples, latents, guides, alpha=alpha)
                running_loss += loss.item()

        valid_losses.append(running_loss / len(valid_loader))

    return tgae, train_losses, valid_losses