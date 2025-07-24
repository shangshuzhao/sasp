import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def embed_ukb_protein(name_list: list) -> list:
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

def match_ukb_dist(df):
    ''' Match the distribution of input dataframe to UK Biobank SASP protein distribution '''
    # UK Biobank SASP protein means and standard deviations
    ukb_means = {
        'ANG': 0.008412209,
        'CCL13': 0.022755328,
        'CCL2': 0.023904332,
        'CCL20': 0.151260181,
        'CCL3': 0.064064144,
        'CCL4': 0.065627897,
        'CHI3L1': 0.133817108,
        'CSF2': 0.009861640,
        'CXCL1': 0.110607086,
        'CXCL10': 0.082652478,
        'CXCL8': 0.042198496,
        'CXCL9': 0.097004956,
        'FGA': 0.012391110,
        'FGF2': 0.078542032,
        'FSTL3': 0.031882412,
        'GDF15': 0.073444923,
        'HGF': 0.032068841,
        'ICAM1': 0.012879968,
        'IGFBP2': -0.050848724,
        'IGFBP6': 0.023432412,
        'IL1B': 0.073486292,
        'IL4': -0.141638803,
        'IL5': 0.268410990,
        'IL6': 0.124677371,
        'IL6ST': 0.001299400,
        'LEP': -0.082768320,
        'MIF': -0.022219210,
        'MMP12': 0.043234173,
        'PGF': 0.020601765,
        'PLAUR': 0.020406817,
        'SERPINE1': -0.041613098,
        'TF': -0.002692966,
        'TIMP1': 0.022978616,
        'TNFRSF11B': 0.014422148,
        'TNFRSF1A': 0.029982814,
        'TNFRSF1B': 0.034870779,
        'TNFSF10': -0.006918623,
        'VEGFA': 0.088493049
    }

    ukb_sds = {
        'ANG': 0.3955384,
        'CCL13': 0.7855913,
        'CCL2': 0.5965727,
        'CCL20': 1.0096516,
        'CCL3': 0.7045312,
        'CCL4': 0.7433986,
        'CHI3L1': 0.9136581,
        'CSF2': 0.3452719,
        'CXCL1': 1.0677707,
        'CXCL10': 0.7815331,
        'CXCL8': 0.7725987,
        'CXCL9': 0.7935531,
        'FGA': 0.3167290,
        'FGF2': 0.7595377,
        'FSTL3': 0.4021560,
        'GDF15': 0.5895939,
        'HGF': 0.4402710,
        'ICAM1': 0.3362013,
        'IGFBP2': 0.8046838,
        'IGFBP6': 0.3838551,
        'IL1B': 0.6282647,
        'IL4': 0.9842809,
        'IL5': 1.5453851,
        'IL6': 0.8839393,
        'IL6ST': 0.1975127,
        'LEP': 1.2912836,
        'MIF': 0.7687430,
        'MMP12': 0.6566276,
        'PGF': 0.3272117,
        'PLAUR': 0.3394527,
        'SERPINE1': 0.7684842,
        'TF': 0.2136098,
        'TIMP1': 0.3305618,
        'TNFRSF11B': 0.3596158,
        'TNFRSF1A': 0.3629309,
        'TNFRSF1B': 0.4259883,
        'TNFSF10': 0.2934764,
        'VEGFA': 0.6615517
    }

    df_transformed = df.copy()

    for col in df.columns:

        if not col in ukb_means:
            raise ValueError(f"Protein {col} not included in the UK Biobank")

        col_mean = df[col].mean()
        col_std = df[col].std()
        target_mean = ukb_means[col]
        target_std = ukb_sds[col]

        df_transformed[col] = (df[col] - col_mean) / col_std
        df_transformed[col] = df_transformed[col] * target_std + target_mean

    return df_transformed

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
    var_label = embed_ukb_protein(var_names)
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

def criterion(recon_x, x, latent, tl, alpha=0.5):
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