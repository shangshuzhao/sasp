import argparse
import numpy as np
import pandas as pd

import torch

from TransformerAutoEncoder import TransformerAE
from protein_to_label import name_to_embed
from match_dist import match_ukb_dist
from rename_medex_protein import rename_medex_columns

def gen_single_index(row, tgae, device):
    """
    Input: single row of protein data as Pandas DF
    Output: single scalar of SASP index
    """
    # Drop missing proteins
    valid_proteins = row.dropna()

    # Get protein names and their values
    protein_names = valid_proteins.index.tolist()
    protein_values = valid_proteins.values.tolist()

    # Convert protein names to indices
    protein_indices = name_to_embed(protein_names)

    # Convert to torch tensors
    protein_idx_tensor = torch.tensor(protein_indices, dtype=torch.long).unsqueeze(0).to(device)   # Shape: (1, P)
    protein_value_tensor = torch.tensor([protein_values], dtype=torch.float32).to(device) # Shape: (1, P)

    tgae.eval()
    with torch.no_grad():
        sasp_index = tgae.predict(protein_value_tensor, protein_idx_tensor)

    sasp_index = sasp_index.cpu().numpy()

    return sasp_index.item()

def gen_index(raw_df, tgae, device):
    """
    Input: protein data as Pandas DF, with missing values
    Output: SASP index as Pandas DF
    """
    # Predefined list of 38 expected proteins
    ALLOWED_PROTEINS = [
        'ANG', 'CCL13', 'CCL2', 'CCL20', 'CCL3', 'CCL4', 'CHI3L1', 'CSF2',
        'CXCL1', 'CXCL10', 'CXCL8', 'CXCL9', 'FGA', 'FGF2', 'FSTL3', 'GDF15',
        'HGF', 'ICAM1', 'IGFBP2', 'IGFBP6', 'IL1B', 'IL4', 'IL5', 'IL6',
        'IL6ST', 'LEP', 'MIF', 'MMP12', 'PGF', 'PLAUR', 'SERPINE1', 'TF',
        'TIMP1', 'TNFRSF11B', 'TNFRSF1A', 'TNFRSF1B', 'TNFSF10', 'VEGFA'
    ]

    invalid_proteins = [col for col in raw_df.columns if col not in ALLOWED_PROTEINS]

    if invalid_proteins:
        raise ValueError(f"The following proteins are not allowed: {invalid_proteins}")

    sasp_index_list = []
    for _, row in raw_df.iterrows():
        output = gen_single_index(row, tgae, device)
        sasp_index_list.append(output)

    return sasp_index_list

def main(alpha, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load TGAE model
    tgae = TransformerAE().to(device)
    model_path = f"gae_a{str(alpha)[2:]}_s{seed}.pth"
    tgae.load_state_dict(torch.load(model_path))

    ukb_raw = pd.read_csv("ukb/ukb_sasp_2.csv")

    # extended sasp index from ukb
    ukb_proteins = ukb_raw.iloc[:,7:45]
    index_1 = gen_index(ukb_proteins, tgae, device)

    # original sasp protein from ukb
    indexes = [0, 1, 3, 5, 7, 9, 10, 11, 14, 19, 22, 23, 24, 25, 26, 27, 29, 31, 32, 34, 35, 37]
    ukb_proteins = ukb_raw.iloc[:,[i + 7 for i in indexes]]
    index_2 = gen_index(ukb_proteins, tgae, device)

    # Combine results
    id = ukb_raw.iloc[:,0]
    index_1 = pd.DataFrame({'sasp_index_e': index_1})
    index_2 = pd.DataFrame({'sasp_index_o': index_2})
    df_combined = pd.concat([id, index_1, index_2], axis=1)
    df_combined.to_csv(f"sasp_ukb_a{str(alpha)[2:]}_s{seed}.csv", index=False)

# --- CONFIGURATION ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    args = parser.parse_args()

    main(alpha=args.alpha, seed=args.seed)
