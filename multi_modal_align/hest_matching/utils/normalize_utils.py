import scipy
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler


def get_normalize_method(normalize_method, **kwargs):
    if normalize_method is None:
        return None
    elif normalize_method == "log1p":
        return log1p
    elif normalize_method == "stdiff":
        return stdiff_normalize
    else:
        raise ValueError(f"Unknown normalize method: {normalize_method}")


def identity(adata):
    return adata.copy()


def scale(adata):
    scaler = MaxAbsScaler()
    normalized_data = scaler.fit_transform(adata.X.T).T
    adata.X = normalized_data
    return adata


def log1p(adata):
    process_data = adata.copy()
    sc.pp.log1p(process_data)
    return process_data


# https://github.com/fdu-wangfeilab/stDiff/blob/master/test-stDiff.py#L47
def stdiff_normalize(adata):
    process_adata = adata.copy()
    sc.pp.normalize_total(process_adata, target_sum=1e4)
    sc.pp.log1p(process_adata)
    process_adata = scale(process_adata)
    if isinstance(process_adata.X, scipy.sparse.csr_matrix):
        process_adata.X.data = process_adata.X.data * 2 - 1
    else:
        process_adata.X = process_adata.X * 2 - 1
    return process_adata


def data_augment(adata, fixed, noise_std):
    augmented_adata = adata.copy()    
    if fixed: 
        augmented_adata.X = augmented_adata.X + np.full(adata.X.shape, noise_std)
    else:
        augmented_adata.X = augmented_adata.X + np.abs(np.random.normal(0, noise_std, adata.X.shape))   
    return adata.concatenate(augmented_adata, join='outer')
