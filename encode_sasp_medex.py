import argparse
import pandas as pd
import torch

import utils
from TransformerAutoEncoder import TransformerAE
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
    protein_indices = utils.embed_ukb_protein(protein_names)

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

def main(args):

    # --- HYPERPARAMETERS ---
    prefix = args.prefix
    bn = args.bn
    alpha = args.alpha
    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load TGAE model and data
    tgae = TransformerAE(latent_dim=bn).to(device)
    m_path = f"gae_{prefix}_b{bn}_a{str(alpha)[2:]}_s{seed}.pth"
    tgae.load_state_dict(torch.load(m_path))

    medex_imputed = pd.read_csv("df_medex/MEDEX_Expanded_SASP_ALL_impute.csv")
    medex_missing = pd.read_csv("df_medex/MEDEX_Expanded_SASP_ALL_missing.csv")

    # extended sasp index using medex imputed
    medex_proteins = medex_imputed.iloc[:,4:]
    medex_proteins = rename_medex_columns(medex_proteins)
    medex_proteins = utils.match_ukb_dist(medex_proteins)
    index_1 = gen_index(medex_proteins, tgae, device)

    # original sasp index using medex imputed
    indexes = [0, 1, 3, 5, 7, 9, 10, 11, 14, 19, 22, 23, 24, 25, 26, 27, 29, 31, 32, 34, 35, 37]
    medex_proteins = medex_imputed.iloc[:,[i + 4 for i in indexes]]
    medex_proteins = rename_medex_columns(medex_proteins)
    medex_proteins = utils.match_ukb_dist(medex_proteins)
    index_2 = gen_index(medex_proteins, tgae, device)

    # extended sasp index using medex raw
    medex_proteins = medex_missing.iloc[:,4:42]
    medex_proteins = rename_medex_columns(medex_proteins)
    medex_proteins = utils.match_ukb_dist(medex_proteins)
    index_3 = gen_index(medex_proteins, tgae, device)

    # original sasp index using medex raw
    indexes = [0, 1, 3, 5, 7, 9, 10, 11, 14, 19, 22, 23, 24, 25, 26, 27, 29, 31, 32, 34, 35, 37]
    medex_proteins = medex_missing.iloc[:,[i + 4 for i in indexes]]
    medex_proteins = rename_medex_columns(medex_proteins)
    medex_proteins = utils.match_ukb_dist(medex_proteins)
    index_4 = gen_index(medex_proteins, tgae, device)

    # Combine results
    id = medex_imputed.iloc[:,0:4]
    index_1 = pd.DataFrame({'sasp_index_e_imputed': index_1})
    index_2 = pd.DataFrame({'sasp_index_r_imputed': index_2})
    index_3 = pd.DataFrame({'sasp_index_e_missing': index_3})
    index_4 = pd.DataFrame({'sasp_index_r_missing': index_4})
    df_combined = pd.concat([id, index_1, index_2, index_3, index_4], axis=1)

    i_path = f"sasp_medex_{prefix}_b{bn}_a{str(alpha)[2:]}_s{seed}.csv"
    df_combined.to_csv(i_path, index=False)


# --- CONFIGURATION ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--bn', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    main(args)
