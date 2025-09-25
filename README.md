# SASP Index (TGAE)

Transformer-based Graph-ish Autoencoder (TGAE) to compute a **Senescence-Associated Secretory Phenotype (SASP) index** from a panel of 38 inflammatory/protein markers. This repository includes:

- A PyTorch model (`TGAE`) for encoding protein profiles and predicting a scalar SASP index
- Utilities for data cleaning and UK Biobank distribution matching
- Helpers to compute the SASP index for new samples
- Simple fine-tuning routines and an example notebook

---

## Repo Structure

```
.
├── example.ipynb           # End-to-end demo: loading data, fine-tuning, and evaluating
├── tgae.py                 # Model: VariableEncoder/Decoder + TGAE
├── proteins_info.py        # Protein list, UKB distribution stats, data cleaning helpers
├── get_index.py            # Batch & single-row SASP index computation helpers
├── fine_tune.py            # Train/valid split and fine-tuning loop utilities
├── utils.py                # Training utilities, losses, embedding helpers, reproducibility
└── tgae_pre_trained.pth    # Pre-trained TGAE checkpoint
```

---

## Installation

Python 3.9+ recommended.

```bash
# install dependencies
pip install torch numpy pandas matplotlib
```

---

## The 38 Required Proteins

Your input must contain **exact all** of the following columns (case-insensitive accepted by the cleaner, but normalized to UPPERCASE internally):

```
ANG, CCL13, CCL2, CCL20, CCL3, CCL4, CHI3L1, CSF2, CXCL1, CXCL10, CXCL8, CXCL9, FGA, FGF2, FSTL3, GDF15, HGF, ICAM1, IGFBP2, IGFBP6, IL1B, IL4, IL5, IL6, IL6ST, LEP, MIF, MMP12, PGF, PLAUR, SERPINE1, TF, TIMP1, TNFRSF11B, TNFRSF1A, TNFRSF1B, TNFSF10, VEGFA
```

---

## Quickstart: Compute SASP Index for New Data

Below is a minimal example showing how to load the pre-trained model and compute the SASP index for each row of a `pandas.DataFrame` that has **exactly** the 38 proteins as columns (order does not matter). Any extra columns should be dropped beforehand.

```python
import torch
import pandas as pd

from tgae import TGAE
from get_index import gen_index

# 1) Prepare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load pre-trained model
tgae = TGAE(d_model=128, latent_dim=6, nhead=8, num_layers=2).to(device)
tgae.load_state_dict(torch.load("tgae_pre_trained.pth", map_location=device))
tgae.eval()

# 3) Load your data (must contain the 38 proteins as columns)
df = pd.read_csv("your_protein_table.csv")

# 4) Compute SASP index per row
sasp = gen_index(df, tgae, device)  # -> list[float] aligned to df rows
print(sasp[:5])
```

**Input expectations:**
- Columns must be the 38 protein names.
- Values should be numeric (float-like). Missing values are handled row-wise by dropping missing proteins for that row before inference.
- The helper will **standardize** your data to match UK Biobank distribution characteristics internally.

If your DataFrame contains other columns (e.g., `ID`, `AGE`, etc.), please select only the protein columns first:

```python
protein_cols = [
    "ANG","CCL13","CCL2","CCL20","CCL3","CCL4","CHI3L1","CSF2",
    "CXCL1","CXCL10","CXCL8","CXCL9","FGA","FGF2","FSTL3","GDF15",
    "HGF","ICAM1","IGFBP2","IGFBP6","IL1B","IL4","IL5","IL6","IL6ST",
    "LEP","MIF","MMP12","PGF","PLAUR","SERPINE1","TF","TIMP1",
    "TNFRSF11B","TNFRSF1A","TNFRSF1B","TNFSF10","VEGFA",
]
df = df[protein_cols]
```

---

## Fine-tuning

You can fine-tune the pre-trained TGAE on a dataset (e.g., to better match a cohort) using the utilities in `fine_tune.py` and `utils.py`. A high-level sketch:

```python
import torch
import pandas as pd

from tgae import TGAE
from fine_tune import split_df, train
from proteins_info import clean_data
import utils

# Load your dataset (may include ID/AGE/etc.)
df = pd.read_csv("your_training_table.csv")

# Optional: normalize and validate columns / get indices
# - `clean_data` checks that all 38 proteins are present (case-insensitive) and returns their positions.
protein_idx, age_idx = clean_data(df, tunning=True)  # age_idx used if your training targets involve age

# Split into train/valid (by rows)
train_idx, valid_idx = split_df(len(df), train_size=0.6, valid_size=0.2)
train_df = df.iloc[train_idx]
valid_df = df.iloc[valid_idx]

# Build model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tgae = TGAE(d_model=128, latent_dim=6, nhead=8, num_layers=2).to(device)

# Prepare PyTorch DataLoaders & embeddings
train_loader, valid_loader, proteins_embedding = utils.prepare_data(
    train_sample=train_df.iloc[:, protein_idx],
    train_target=train_df.iloc[:, age_idx] if age_idx != -1 else None,
    valid_sample=valid_df.iloc[:, protein_idx],
    valid_target=valid_df.iloc[:, age_idx] if age_idx != -1 else None,
    device=device,
)

# Optimizer / epochs / alpha (reconstruction vs guidance loss)
optimizer = torch.optim.Adam(tgae.parameters(), lr=1e-3, weight_decay=1e-5)
tgae, train_losses, valid_losses = train(
    tgae=tgae,
    proteins_embedding=proteins_embedding,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    epochs=50,
    alpha=0.5,  # tune to balance reconstruction vs. guidance losses
)
```

> See `example.ipynb` for a full, runnable notebook that loads CSVs, fine-tunes, and evaluates correlations (e.g., with age).

---

## API Overview

### `tgae.TGAE`
- **`forward(x, var_label)`** → `(latent, recon)`  
  - `x`: `(batch, P)` protein values  
  - `var_label`: `(1, P)` integer embedding of variable names  
  - Returns a **latent scalar prediction** (the SASP index) and a **reconstruction** of inputs.
- **`predict(x, var_label)`** → `latent`  
  - Convenience method to get only the SASP index.

### `proteins_info.py`
- `protein_list`: canonical ordered tuple of the 38 proteins.  
- `match_ukb_dist(df)`: standardizes each protein column of `df` to match UK Biobank means/SDs.  
- `clean_data(df, tunning=False, proteins_list=protein_list)`: validates presence of required proteins (case-insensitive); optionally finds the `age` column when `tunning=True`.

### `get_index.py`
- `gen_single_index(row, tgae, device)`: compute SASP index for a single row.  
- `gen_index(df, tgae, device)`: compute SASP index for all rows (returns `list[float]`).

### `utils.py`
- Embedding helpers to map protein names to integer IDs for the transformer.
- Reproducibility helpers, data preparation, loss routines, and a simple training loop.

### `fine_tune.py`
- `split_df(n_rows, train_size, valid_size)` → train/valid/test indices  
- `train(...)` → fine-tuning loop wrapper

---

## Tips & Gotchas

- **Column names**: Use the exact protein names above. `clean_data` will raise if names are missing or duplicated in a case-insensitive way.
- **Standardization**: You normally don’t need to standardize beforehand—the helper aligns to UKB distribution—but ensure your input scale is reasonable (floats).
- **Missing values**: For inference, missing per-row values are dropped for that row. For training, ensure consistent preprocessing.
- **Determinism**: Set seeds to make splits and training reproducible (see `utils`).

---

## Example Notebook

Open `example.ipynb` for a complete walkthrough:
- Set hyperparameters
- Load & clean data
- Fine-tune the model
- Compute and inspect SASP index

---

## Citation / License

If you use this code in academic work, please cite appropriately.  
**License**: _Fill in your preferred license here (e.g., MIT, Apache-2.0)._


---

## Acknowledgements

Thanks to UK Biobank–based statistics included in `proteins_info.py` for distribution matching.

