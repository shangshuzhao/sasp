def match_ukb_dist(df):
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
