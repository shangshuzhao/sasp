import argparse
import pandas as pd
import torch

import utils
from TransformerAutoEncoder import TransformerAE

def main(args):

    # --- HYPERPARAMETERS ---
    prefix = args.prefix
    bn = args.bn
    alpha = args.alpha
    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    utils.set_seed(seed)

    # --- IMPORT DATA ---

    ukb_sasp_train = pd.read_csv("df_ukb/ukb_sasp_train.csv")
    ukb_sasp_valid = pd.read_csv("df_ukb/ukb_sasp_val.csv")

    ukb_train_sample = ukb_sasp_train.iloc[:,7:]
    ukb_train_target = ukb_sasp_train.iloc[:,5]

    ukb_valid_sample = ukb_sasp_valid.iloc[:,7:]
    ukb_valid_target = ukb_sasp_valid.iloc[:,5]

    ukb_train_loader, ukb_valid_loader, proteins_label = utils.prepare_data(
        ukb_train_sample, ukb_train_target, ukb_valid_sample, ukb_valid_target, device)

    # --- INSTANTIATE ---

    tgae = TransformerAE(latent_dim=bn).to(device)

    optimizer = torch.optim.Adam(
        tgae.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    # --- TRAIN LOOP ---
    num_epochs = 200
    tgae, train_losses, valid_losses = utils.training_sasp(
        num_epochs, tgae, optimizer, ukb_train_loader, ukb_valid_loader, proteins_label, alpha=alpha)
    
    # --- SAVE TRAINED MODEL ---
    m_path = f"gae_{prefix}_b{bn}_a{str(alpha)[2:]}_s{seed}.pth"
    torch.save(tgae.state_dict(), m_path)

    # --- PLOT LOSS ---
    p_path = f"loss_{prefix}_b{bn}_a{str(alpha)[2:]}_s{seed}.png"
    utils.plot_losses(train_losses, valid_losses, p_path)

# --- CONFIGURATION ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--bn', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    main(args)
