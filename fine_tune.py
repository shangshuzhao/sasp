import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import utils

def split_df(n_rows, train_size, valid_size):
    # pick sizes (rounded to nearest int), capped so they fit
    n60 = math.floor(n_rows * train_size)
    n80 = math.floor(n_rows * (train_size + valid_size))

    # reproducible random permutation of row positions
    rng = np.random.default_rng(42)  # change seed as you like
    perm = rng.permutation(n_rows)

    # split positions, then map back to the dataframe's index labels
    pos_60 = perm[:n60]
    pos_20 = perm[n60:n80]

    return pos_60, pos_20

def prepare_data(train_sample, train_target, device):
    '''
    Prepare dataloader and protein name embeddings 
    '''

    # Create a copy of protein name embeddings
    var_names = train_sample.columns.tolist()
    var_label = utils.embed_protein(var_names)
    var_label = torch.tensor(var_label).to(device)

    # Convert to NumPy arrays
    train_sample = train_sample.values.astype(np.float32)
    train_target = train_target.values.astype(np.float32)

    # Create TensorDataset and DataLoader
    data = TensorDataset(
        torch.tensor(train_sample).to(device),
        torch.tensor(train_target).to(device).squeeze(1)
    )
    loader = DataLoader(data, batch_size=128, shuffle=True)

    return loader, var_label

def train_one_epoch(tgae, protein_embedding, data_loader, optimizer):

    tgae.train()
    running_loss = 0.0

    for samples, guides in data_loader:
        latents, recon_x = tgae(samples, protein_embedding)

        loss = utils.criterion(recon_x, samples, latents, guides)
        loss.backward()
        running_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

    return tgae, optimizer, running_loss / len(data_loader)

def valid_one_epoch(tgae, protein_embedding, valid_loader, optimizer):

    tgae.eval()
    running_loss = 0.0

    with torch.no_grad():
        for samples, guides in valid_loader:
            latents, recon_x = tgae(samples, protein_embedding)

            loss = utils.criterion(recon_x, samples, latents, guides)
            running_loss += loss.item()

    return tgae, optimizer, running_loss / len(valid_loader)

def fine_tune(
        epochs,
        tgae,
        proteins_embedding,
        train_loader,
        valid_loader,
        optimizer,
):
    train_losses = []
    valid_losses = []

    for i in range(epochs):
        print("Training epoch: ", i+1)
        tgae, optimizer, train_loss = train_one_epoch(tgae, proteins_embedding, train_loader, optimizer)
        train_losses.append(train_loss)

        tgae, optimizer, valid_loss = train_one_epoch(tgae, proteins_embedding, valid_loader, optimizer)
        valid_losses.append(valid_loss)

    return tgae, train_losses, valid_losses