import numpy as np
import pandas as pd
import os
import math

def sorting_by_cutoff(df,source):
    """
    Filters variants by mutation count and read-count availability, and optionally
    ranks them by total counts in the pre-sort library and a selected gate.

    :param df: pd.DataFrame - DataFrame containing variant counts and mutation information
    :param source: str - data source used for filtering; 'presort' filters by pre-sort
        counts only, while any other value is treated as a gate name and used to filter
        by the corresponding gate count column
    :return: pd.DataFrame - filtered DataFrame containing variants with up to two
        mutations and nonzero counts in the selected source; for gate-specific sources,
        the returned DataFrame is additionally sorted in descending order by the sum of
        pre-sort and gate counts
    """
    if source=='presort':
        rel_index = (df['count pre'] > 0)&(df['number of mutations']<=2)
        filtered_df = df.loc[rel_index].copy()
        return filtered_df
    else:
        gate_name = 'count ' + source
        gate_sum = f"{source}_sum"
        rel_index=(df[gate_name] > 0)&(df['number of mutations']<=2)
        filtered_df = df.loc[rel_index].copy()
        filtered_df[gate_sum]=filtered_df[gate_name]+filtered_df['count pre']
        sorted_df = filtered_df.sort_values(by=gate_sum, ascending=False)

        return sorted_df

def pearson_cal(x, y):
    """
      Computes the Pearson correlation coefficient between two arrays.

      :param x: array - first data series
      :param y: array - second data series
      :return: float - Pearson correlation coefficient
      """
    # Calculate Pearson correlation coefficient
    correlation_matrix = np.corrcoef(x, y)
    return round(correlation_matrix[0, 1], 3)

def set_random_seeds(seed_value):
    """
    Sets random seeds for TensorFlow, NumPy, and Python's random module to ensure reproducibility.

    :param seed_value: int - seed value for reproducibility
    """
    import random
    import tensorflow as tf

    # Set random seeds for reproducibility
    np.random.seed(seed_value)
    tf.keras.utils.set_random_seed(seed_value)
    random.seed(seed_value)

def shuffle_data(index, ER, X_extra, seed):
    """
    Shuffles ESM input features, target values, and one-hot encoded input features
    using a fixed random seed.

    :param index: np.ndarray - ESM representation input array
    :param ER: np.ndarray - target values associated with the samples
    :param X_extra: np.ndarray - one-hot encoded input array associated with the samples
    :param seed: int - random seed used for reproducible shuffling
    :return: tuple - shuffled ESM input array, shuffled target values, and shuffled
        one-hot encoded input array
    """
    # Set seeds for reproducibility
    set_random_seeds(seed)

    # shuffle the train parameters: input, value, and weights
    shuffle_idx = np.random.permutation(len(index))
    shuffle_index = index[shuffle_idx]
    shuffle_value = ER[shuffle_idx]
    shuffle_extra = X_extra[shuffle_idx]

    return shuffle_index, shuffle_value, shuffle_extra

def enrichment_ratio_cal(df_count_cut_pre, df_count_cut_gate, WT, seqs_in_gate, seqs_in_pre,gate):
    """
       Calculates log2 enrichment ratios for variants in a selected gate relative to the pre-sort library.

       :param df_count_cut_pre: DataFrame - filtered variant counts from the pre-sort library
       :param df_count_cut_gate: DataFrame - filtered variant counts from the selected gate
       :param WT: str - wild-type sequence used for normalization
       :param seqs_in_gate: list - sequences observed in the selected gate
       :param seqs_in_pre: list - sequences observed in the pre-sort library
       :param gate: str - name of the target gate
       :return: tuple - enrichment values, WT reference lists, gate counts, and updated gate DataFrame
       """
    enrichment_gate = {}
    freq_mut_after = {}
    freq_mut_pre = {}
    counter_after_enrich={}
    gate_name='count '+ gate
    X = []
    Y = []
    freq_mut_after[WT] = (df_count_cut_gate.loc[WT,gate_name]) / len(seqs_in_gate)
    freq_mut_pre[WT] = (df_count_cut_pre.loc[WT, 'count pre']) / len(seqs_in_pre)
    enrichment_gate[WT] = math.log2(
        ((freq_mut_after[WT]) / (freq_mut_after[WT])) / ((freq_mut_pre[WT]) / (freq_mut_pre[WT])))
    X.append(WT)
    Y.append(enrichment_gate[WT])
    counter_after_enrich[WT]=df_count_cut_gate.loc[WT,gate_name]
    df_count_cut_gate.loc[WT, f'ER {gate}'] = enrichment_gate[WT]
    for key in df_count_cut_pre.index:
        if key in df_count_cut_gate.index and key != WT:
            freq_mut_after[key] = (df_count_cut_gate.loc[key, gate_name]) / len(seqs_in_gate)
            freq_mut_pre[key] = (df_count_cut_pre.loc[key, 'count pre']) / len(seqs_in_pre)
            enrichment_gate[key] = math.log2(
                ((freq_mut_after[key]) / (freq_mut_after[WT])) / ((freq_mut_pre[key]) / (freq_mut_pre[WT])))

            counter_after_enrich[key]=((df_count_cut_gate.loc[key, gate_name]))
            df_count_cut_gate.loc[key,f'ER {gate}']=enrichment_gate[key]

    return enrichment_gate, X, Y,counter_after_enrich,df_count_cut_gate

def prepare_data_for_model(enrich_gate,df_count_cut_gate, gate,X_all_pos,choice,target_position=None):
    """
    Prepares training and test data for model fitting based on the selected split scheme.

    :param enrich_gate: dict - enrichment-ratio values for variants in the selected gate
    :param df_count_cut_gate: DataFrame - filtered variant DataFrame for the selected gate
    :param gate: str - name of the target gate
    :param X_all_pos: list - list of all variant sequences
    :param choice: int - training scheme selector
    :param target_position: str or int or None - target mutation position for choice 5
    :return: tuple - test sequences, ESM training input, one-hot training input,
        ESM test input, one-hot test input, training targets, and test targets
    """
    import pandas as pd

    if choice == 1: # train 90% test the remaining 10%
        size = int(0.1 * len(df_count_cut_gate))
        test_df = df_count_cut_gate[:size]
        train_df = df_count_cut_gate[size:]
        gate_name = 'count ' + gate

        X_want_test, y_want_test = test_df.index, test_df[f'ER {gate}'].values
        X_all_train, y_all_train = train_df.index, train_df[f'ER {gate}'].values
    elif choice == 2: # 100% of the data using for training

        X_all_train = []
        y_all_train = []
        y_want_test = []
        X_want_test = []

        for string in X_all_pos:
            X_want_test.append((string))
            if string in enrich_gate:
                X_all_train.append(string)
                y_all_train.append(enrich_gate[string])

    elif choice == 3:
        # 27 single points
        test_dic = {'AGPKARIVYGGR': 0.22, 'TAPKARIVYGGR': 0.68, 'TGAKARIVYGGR': -0.06, 'TGPAARIVYGGR': 1.99,
                    'TGPKAAIVYGGR': 0.55, 'TGPKARAVYGGR': 1.4, 'TGPKARIAYGGR': 0.05, 'TGPKARIVAGGR': 0.88,
                    'TGPKARIVYAGR': 0.95,
                    'TGPKARIVYGAR': 0.81, 'TGPKARIVYGGA': 0.22, 'TGPGARIVYGGR': 4.11, 'TGPSARIVYGGR': 3.39,
                    'TGPVARIVYGGR': 2.13,
                    'TGPTARIVYGGR': 2.09, 'TGPDARIVYGGR': 5.22, 'TGPNARIVYGGR': 1.32, 'TGPMARIVYGGR': -1.42,
                    'TGPIARIVYGGR': 2.93,
                    'TGPLARIVYGGR': -1.58, 'TGPEARIVYGGR': 3.87, 'TGPQARIVYGGR': 0.31, 'TGPHARIVYGGR': 0.1,
                    'TGPRARIVYGGR': -0.61,
                    'TGPFARIVYGGR': -1.96, 'TGPYARIVYGGR': -2.61, 'TGPWARIVYGGR': -2.44}
        # test set
        X_want_test = list(test_dic.keys())
        y_want_test = df_count_cut_gate[f'ER {gate}'].reindex(X_want_test).to_numpy()

        # train set = all other variants
        train_mask = ~df_count_cut_gate.index.isin(X_want_test)
        X_all_train = df_count_cut_gate.index[train_mask].to_numpy()
        y_all_train = df_count_cut_gate.loc[train_mask, f'ER {gate}'].to_numpy()
    elif choice == 4:
        # 10 double points
        test_dic = {'TGPRGRIVYGGR': 0.83072472, 'TGPRSRIVYGGR': 1.501384323, 'TGPRRRIVYGGR': 4.045516184,
                    'TGPRVRIVYGGR': 6.002537461, 'TGPRLRIVYGGR': 4.921405335, 'TGPRWRIVYGGR': 6.63304497,
                    'TGPYARLVYGGR': -3.49201313,
                    'TGARARIVYGGR': -0.588160841, 'TGPRAAIVYGGR': -0.4195144, 'TGPRARIAYGGR': -0.537152591}

        # spliting to test,validation and train set

        # test set
        X_want_test = list(test_dic.keys())
        y_want_test = df_count_cut_gate[f'ER {gate}'].reindex(X_want_test).to_numpy()

        # train set = all other variants
        train_mask = ~df_count_cut_gate.index.isin(X_want_test)
        X_all_train = df_count_cut_gate.index[train_mask].to_numpy()
        y_all_train = df_count_cut_gate.loc[train_mask, f'ER {gate}'].to_numpy()

    elif choice ==5:

        # Extract positions for single and double mutations
        def extract_positions(mutation):
            if '_' in mutation:
                # Double mutation, split and return both positions
                return mutation.split('_')
            else:
                # Single mutation, return the position only
                return [mutation]

        # Apply the extraction function
        df_count_cut_gate['positions'] = df_count_cut_gate['mutations'].apply(extract_positions)

        # Now you can split the data based on these positions
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        # Define the target mutation position
        test_position = str(target_position)  # Look for mutations at specific position

        # Split the data such that mutations involving position  go to the test set
        for index, row in df_count_cut_gate.iterrows():
            # Check if 'position' is in any of the positions of the mutation (single or double)
            if any(test_position in pos for pos in row['positions']):
                test_df = pd.concat([test_df, row.to_frame().T])  # Add to test set
                test_df[f'ER {gate}'] = pd.to_numeric(test_df[f'ER {gate}'], errors='coerce')

            else:
                train_df = pd.concat([train_df, row.to_frame().T])  # Add to train set
                train_df[f'ER {gate}'] = pd.to_numeric(train_df[f'ER {gate}'], errors='coerce')

        X_want_test, y_want_test = test_df.index, test_df[f'ER {gate}'].values
        X_all_train, y_all_train = train_df.index, train_df[f'ER {gate}'].values
    #print('len train', len(X_all_train), 'len test', len(X_want_test))

    X_all = np.load("./X_esm_delta_mean_all_pos_640_par.npy")  # shape (N, D)
    ids_all = np.load( "./variant_ids_all_pos.npy", allow_pickle=True)

    # put into DataFrame so we can subset by index safely
    X_df = pd.DataFrame(X_all, index=ids_all)  # inde
    x_train = X_df.loc[X_all_train].values  # (N_train, D)
    x_test = X_df.loc[X_want_test].values  # (N_test, D)

    # Geenrate one-hot-encoding vectors for each seq
    amino_acids = ["R", "K", "D", "E", "H", "N", "Q", "S", "T", "P", "C", "G", "A", "V", "I", "L", "M", "F", "Y", "W"]
    mapping = {aa: [int(i == j) for j in range(len(amino_acids))] for i, aa in enumerate(amino_acids)}

    def onehot_encode(seq, mapping):
        return np.array([mapping.get(i) for i in seq])

    x_train_one = np.array(pd.Series(X_all_train).apply(lambda seq: onehot_encode(seq, mapping)).tolist())
    x_test_one = np.array(pd.Series(X_want_test).apply(lambda seq: onehot_encode(seq, mapping)).tolist())

    y_train = np.array(y_all_train)
    y_test = np.array(y_want_test)
    return X_want_test,x_train,x_train_one, x_test,x_test_one, y_train, y_test


def create_model(arch, shape=(640,), DO=0):
    """
    Creates a neural network model that combines ESM embeddings and one-hot encoded sequence inputs.

    :param arch: list - number of neurons in each hidden layer
    :param shape: tuple - shape of the ESM input representation
    :param DO: float - dropout rate applied after each hidden layer
    :return: tensorflow.keras.Model - compiled model architecture with ESM and one-hot inputs
    """
    import tensorflow as tf
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Flatten, Dense, Dropout, Concatenate
    onehot_in = Input(shape=(12,20), name="onehot")   # (12,20)
    esm_in  = Input(shape=shape, name="extra")     # (3,)

    x = Flatten()(onehot_in)                        # (240,)
    x = Concatenate()([esm_in,x])                # (243,)

    for neurons in arch:
        x = Dense(neurons, activation='relu')(x)
        if DO != 0:
            x = Dropout(DO)(x)  # assuming DO is a float like 0.2

    out = Dense(1)(x)  # adjust if classification etc.
    return Model([esm_in,onehot_in], out)

