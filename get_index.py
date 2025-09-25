import numpy as np
import torch
import utils

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
    protein_indices = utils.embed_protein(protein_names)

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
    
    raw_df = raw_df.astype(np.float32)
    std_df = utils.match_ukb_dist(raw_df)

    sasp_index_list = []
    for _, row in std_df.iterrows():
        output = gen_single_index(row, tgae, device)
        sasp_index_list.append(output)

    return sasp_index_list
