import pandas as pd

def rename_medex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames specific columns in the input DataFrame and converts column names 
    from position 5 to 42 (0-based index 4 to 41) to uppercase.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Modified DataFrame with renamed and uppercased columns
    """
    # Column rename mapping
    rename_dict = {
        "GP130": "IL6ST",
        "IL8": "CXCL8",
        "YKL40": "CHI3L1",
        "VEGF": "VEGFA",
        "GMCSF": "CSF2",
        "CFIII": "TF",
        "TNFR1": "TNFRSF1A",
        "TRAIL": "TNFSF10",
        "TNFR2": "TNFRSF1B",
        "PLGF": "PGF",
        "Angiogenin": "ANG",
        "Osteoprotegerin": "TNFRSF11B",
        "uPAR": "PLAUR",
        "DDimer": "FGA",
        "Leptin": "LEP"
    }

    # Apply specific renames
    df = df.rename(columns=rename_dict)
    df = df.rename(columns={col: col.upper() for col in df.columns})

    return df
