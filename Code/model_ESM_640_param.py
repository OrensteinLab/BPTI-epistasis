import random
import tensorflow as tf
import pandas as pd
import statistics
import math
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import os
from keras import optimizers
import itertools

from tensorflow.python.keras.legacy_tf_layers.core import dropout


data_path="./"



# Read the Excel with all the mutant variants possible (including wt, single and double)
curr_all_pos_data_path=os.path.join(data_path,'df_all_variant_0-2_mutations_long.csv')
df_all_variants=pd.read_csv(curr_all_pos_data_path, index_col=0)
X_all_pos=list(df_all_variants['mutated_sequence'])
X_all_pos_long=list(df_all_variants['long_mut_seq'])


def sorting_by_cutoff(df,source):
    if source=='presort':
        rel_index = (df['count pre'] > 0)&(df['number of mutations']<=2)
        filtered_df = df.loc[rel_index].copy()
        #filtered_df = df[(df['count pre'] > 0)&(df['number of mutations']<=2)]
        return filtered_df
    else:
        gate_name = 'count ' + source
        gate_sum = f"{source}_sum"
        rel_index=(df[gate_name] > 0)&(df['number of mutations']<=2)
        filtered_df = df.loc[rel_index].copy()
        #filtered_df = df[(df[gate_name] > 0)&(df['number of mutations']<=2)]
        filtered_df[gate_sum]=filtered_df[gate_name]+filtered_df['count pre']
        sorted_df = filtered_df.sort_values(by=gate_sum, ascending=False)

        return sorted_df


WT_pro_seq='TGPKARIVYGGR'
WT_pro_seq_long='RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRT'


import os
import numpy as np
import pandas as pd
import torch
import esm

# --- YOU MUST SET THIS ---
LONG_COL = "long seq"

# WT full-length sequence (string)
wt_full_seq = WT_pro_seq_long

# Mutated positions (1-based)
POSITIONS_1BASED = [11,12, 13, 15, 16, 17, 18, 34, 35,36,37,39]
POS0 = [p - 1 for p in POSITIONS_1BASED]


mutant_full_seqs=X_all_pos_long
variant_ids=X_all_pos
device = "cpu"

# Small model recommended on CPU; you can swap to 150M later
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model = model.to(device).eval()
batch_converter = alphabet.get_batch_converter()

LAYER = model.num_layers
D = model.embed_dim
print("layers:", LAYER, "embed_dim:", D)

@torch.inference_mode()
def selected_pos_embeddings_batch(seqs, batch_size=256):
    """
    Returns embeddings at your selected positions only.
    Output: (N, 12, D)
    """
    out_all = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        data = [(str(i+j), s) for j, s in enumerate(batch)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        out = model(tokens, repr_layers=[LAYER], return_contacts=False)
        reps = out["representations"][LAYER]  # (B, Lmax+2, D), padded

        for b, s in enumerate(batch):
            L = len(s)
            rep = reps[b, 1:L+1, :]       # (L, D) strip BOS/EOS
            sel = rep[POS0, :]            # (12, D)
            out_all.append(sel.cpu().numpy())

    return np.stack(out_all, axis=0)      # (N, 12, D)

def compute_delta_features_all(wt_seq, mut_seqs, mode="mean", batch_size=256):
    """
     Computes mutation-dependent delta embedding features relative to the wild-type sequence.

     :param wt_seq: str - wild-type protein sequence
     :param mut_seqs: list - list of mutated protein sequences
     :param mode: str - method used to summarize delta embeddings
     :param batch_size: int - batch size for embedding extraction
     :return: np.ndarray - delta feature matrix for the mutated sequences
     """
    wt_sel = selected_pos_embeddings_batch([wt_seq], batch_size=1)[0]  # (12, D)
    mut_sel = selected_pos_embeddings_batch(mut_seqs, batch_size=batch_size)  # (N, 12, D)
    delta = mut_sel - wt_sel[None, :, :]  # (N, 12, D)

    if mode == "mean":
        X = delta.mean(axis=1)  # (N, D)
        return X
    elif mode == "mean_max":
        mean_vec = delta.mean(axis=1)  # (N, D)
        max_vec = delta.max(axis=1)  # (N, D)
        X = np.concatenate([mean_vec, max_vec], axis=1)  # (N, 2D)
        return X
    elif mode == "mean_abs_max":
        mean_vec = delta.mean(axis=1)  # (N, D)
        amax_vec = np.abs(delta).max(axis=1)  # (N, D)
        X = np.concatenate([mean_vec, amax_vec], axis=1)
        return X
    elif mode == "mean_max_min":
        mean_vec = delta.mean(axis=1)  # (N, D)
        max_vec = delta.max(axis=1)  # (N, D)
        min_vec = delta.min(axis=1)  # (N, D)
        X = np.concatenate([mean_vec, max_vec, min_vec], axis=1)  # (N, 3D)n        elif mode == "concat":
        return X
    elif mode == "concat":
        X = delta.reshape(delta.shape[0], -1)  # (N, P*D)
        return X
    else:
        raise ValueError("mode must be 'mean' or 'concat'")

X_esm = compute_delta_features_all(wt_full_seq, mutant_full_seqs, mode="mean", batch_size=256)

out_dir = data_path  # or wherever you want
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, "X_esm_delta_mean_all_pos_640_par.npy"), X_esm)
np.save(os.path.join(out_dir, "variant_ids_all_pos.npy"), np.array(variant_ids, dtype=object))

print("Saved features:", X_esm.shape)

