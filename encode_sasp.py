import argparse
import numpy as np
import pandas as pd

import torch

from TransformerAutoEncoder import TransformerAE
from protein_to_label import name_to_embed
from match_dist import match_ukb_dist

def generate_index(tgae, proteins_label, protein_data):
    """
    Input: model, tensorized protein data
    Output: encoded index as Pandas DF
    """
    tgae.eval()
    with torch.no_grad():
        sasp_index = tgae.predict(protein_data, proteins_label)

    sasp_index = sasp_index.cpu().numpy()
    df = pd.DataFrame(sasp_index, columns=['sasp_index'])
    return df

def process_data(raw_data, device):
    """
    Input: protein raw data as Pandas DF
    Output: encoded protein names, tensorized protein data
    """
    var_names = raw_data.columns.tolist()
    var_label = name_to_embed(var_names)
    var_label = torch.tensor(var_label).unsqueeze(0).to(device)

    np_data = raw_data.values.astype(np.float32)
    tensor_data = torch.tensor(np_data).to(device)
    return var_label, tensor_data

def main(alpha, seed, device):

    # Load TGAE model
    tgae = TransformerAE().to(device)
    model_path = f"gae_a{alpha}_s{seed}.pth"
    tgae.load_state_dict(torch.load(model_path))

    # extended sasp protein from ukb
    ukb_raw = pd.read_csv("ukb/ukb_sasp_2.csv")
    ukb_proteins = ukb_raw.iloc[:,7:45]

    protein_labels, sasp_data = process_data(ukb_proteins, device)
    index = generate_index(tgae, protein_labels, sasp_data)

    id = ukb_raw.iloc[:,0]
    df_combined = pd.concat([id, index], axis=1)
    df_combined.to_csv(f"index_ukb_e_a{alpha}_s{seed}.csv", index=False)

    # original sasp protein from ukb
    ukb_raw = pd.read_csv("ukb/ukb_sasp_2.csv")
    ukb_proteins = ukb_raw.iloc[:,[7, 8, 9, 10, 12, 14, 17, 18, 21, 26, 28, 29, 31, 32, 33, 34, 38, 42, 43, 44]]

    protein_labels, sasp_data = process_data(ukb_proteins, device)
    index = generate_index(tgae, protein_labels, sasp_data)

    id = ukb_raw.iloc[:,0]
    df_combined = pd.concat([id, index], axis=1)
    df_combined.to_csv(f"index_ukb_o_a{alpha}_s{seed}.csv", index=False)

    # extended sasp protein from ukb
    medex_raw = pd.read_csv("medex/medex_renamed.csv")
    medex_proteins = medex_raw.iloc[:,4:]
    medex_proteins = match_ukb_dist(medex_proteins)

    protein_labels, sasp_data = process_data(medex_proteins, device)
    index = generate_index(tgae, protein_labels, sasp_data)

    index.to_csv(f"index_medex_e_a{alpha}_s{seed}.csv", index=False)

    # original sasp protein from ukb
    medex_raw = pd.read_csv("medex/medex_renamed.csv")
    medex_proteins = medex_raw.iloc[:,[4, 5, 6, 7, 9, 11, 14, 15, 18, 23, 25, 26, 28, 29, 30, 31, 35, 39, 40, 41]]
    medex_proteins = match_ukb_dist(medex_proteins)

    protein_labels, sasp_data = process_data(medex_proteins, device)
    index = generate_index(tgae, protein_labels, sasp_data)

    index.to_csv(f"index_medex_o_a{alpha}_s{seed}.csv", index=False)

# --- CONFIGURATION ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(alpha=args.alpha, seed=args.seed, device=device)
