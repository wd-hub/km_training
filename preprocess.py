import numpy as np
from tqdm import tqdm

def normalize_data(data, mean, std):
    normalized_data = []
    pbar = tqdm(data)
    for sample in pbar:
        pbar.set_description('Normalizing data')
        sample_norm = (sample - mean) / std
        normalized_data.append(sample_norm)
    return np.asarray(normalized_data)

def normalize_batch(data):
    flat = np.reshape(data, [len(data), -1])
    mp = np.mean(flat, axis=1)
    sp = np.std(flat, axis=1) + 1e-7
    _mp = np.repeat(np.repeat(np.reshape(mp, [-1, 1, 1, 1]), 32, axis=1), 32, axis=2)
    _sp = np.repeat(np.repeat(np.reshape(sp, [-1, 1, 1, 1]), 32, axis=1), 32, axis=2)
    normalized_data = (data - _mp) / _sp
    return normalized_data

def normalize_depth(data):
    patches_cv = data[:, 16, 16, 0]  # center value of each patch
    patches_all = data - np.tile(np.reshape(patches_cv, (-1, 1, 1, 1)), (1, 32, 32, 1))
    patches_all = np.clip(patches_all, -500, 500)
    patches_all = patches_all / 500.
    return patches_all