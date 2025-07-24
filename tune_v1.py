import argparse
import pandas as pd
import torch

import utils
from TransformerAutoEncoder import TransformerAE
from rename_medex_protein import rename_medex_columns

def main(args):

    # --- HYPERPARAMETERS ---
    prefix = args.prefix
    bn = args.bn
    alpha = args.alpha
    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed(seed)

    # --- IMPORT DATA ---
    medex = pd.read_csv("df_medex/MEDEX_Expanded_SASP_ALL_impute_age.csv")
    medex_baseline = medex[medex['Timepoint'] == 0]

    medex_train_target = medex_baseline.iloc[:400, 42]
    medex_valid_target = medex_baseline.iloc[400:, 42]

    medex_train_sample = medex_baseline.iloc[:400,4:42]
    medex_valid_sample = medex_baseline.iloc[400:,4:42]
    medex_train_sample = rename_medex_columns(medex_train_sample)
    medex_valid_sample = rename_medex_columns(medex_valid_sample)
    medex_train_sample = utils.match_ukb_dist(medex_train_sample)
    medex_valid_sample = utils.match_ukb_dist(medex_valid_sample)

    medex_train_loader, medex_valid_loader, proteins_label = utils.prepare_data(
        medex_train_sample, medex_train_target, medex_valid_sample, medex_valid_target, device)

    # --- NETWORK INSTANTIATE ---

    tgae = TransformerAE().to(device)
    model_path = f"gae_age_b{bn}_a{str(alpha)[2:]}_s{seed}.pth"
    tgae.load_state_dict(torch.load(model_path))

    # Freeze all parameters
    for param in tgae.parameters():
        param.requires_grad = False

    # Unfreeze only the regressor
    for param in tgae.regressor.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(tgae.regressor.parameters(), lr=1e-3,  weight_decay=1e-5)

    # --- TRAIN LOOP ---

    train_losses = []
    valid_losses = []

    num_epochs = 30

    for _ in range(num_epochs):

        tgae.train()
        running_loss = 0.0

        for samples, guides in medex_train_loader:
            latents, recon_x = tgae(samples, proteins_label)

            loss = utils.criterion(recon_x, samples, latents, guides, alpha=alpha)
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

                loss = utils.criterion(recon_x, samples, latents, guides, alpha=alpha)
                running_loss += loss.item()

        valid_losses.append(running_loss / len(medex_valid_loader))

    # --- SAVE TRAINED MODEL ---
    m_filename = f"gae_{prefix}_b{bn}_a{str(alpha)[2:]}_s{seed}.pth"
    torch.save(tgae.state_dict(), m_filename)

    # --- PLOT LOSS ---
    p_filename = f"loss_{prefix}_b{bn}_a{str(alpha)[2:]}_s{seed}.png"
    utils.plot_losses(train_losses, valid_losses, p_filename)

# --- CONFIGURATION ---

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--bn', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    main(args)
