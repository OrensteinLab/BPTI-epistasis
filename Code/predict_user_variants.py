import numpy as np
import pandas as pd
import os
from datetime import datetime
from tensorflow.keras.models import load_model


def read_fasta_to_df(fasta_file):
    """
        Reads a FASTA file and creates a DataFrame with sequence IDs and amino acid sequences.

        :param fasta_file: str - Path to the FASTA file.
        :return: pd.DataFrame - DataFrame with columns ['ID', 'seq'].
    """
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"Error: {fasta_file} not found.")

    sequences = []
    with open(fasta_file, "r") as file:
        seq_id, sequence = None, []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    sequences.append([seq_id, "".join(sequence)])
                seq_id, sequence = line[1:], []
            else:
                sequence.append(line)
        if seq_id:
            sequences.append([seq_id, "".join(sequence)])

    return pd.DataFrame(sequences, columns=["ID", "seq"])


def load_models_from_paths(models_path, model_groups):
    """
    Loads trained models from specified subdirectories under the given path.

    :param models_path: str - base directory containing model subdirectories
    :param model_groups: list of str - names of subdirectories containing models
    :return: list of list - nested list where each inner list contains loaded models
    """

    models = []
    for group in model_groups:
        group_path = os.path.join(models_path, group)
        curr_models = [load_model(os.path.join(group_path, model)) for model in os.listdir(group_path)]
        models.append(curr_models)

    return models


def generate_test_input(test_list):
    """
    Generates model input arrays for a specified list of test sequences.

    :param test_list: list - list of mutated sequences to use as the test set
    :return: tuple - contains:
        X_want_test (list): test sequences,
        x_test (np.ndarray): ESM embedding vectors for the test sequences,
        x_test_one (np.ndarray): one-hot encoded representations of the test sequences
    """

    X_want_test = test_list + ['TGPKARIVYGGR']
    X_all = np.load("./X_esm_delta_mean_all_pos_640_par.npy")  # shape (N, D)
    ids_all = np.load( "./variant_ids_all_pos.npy", allow_pickle=True)

    # put into DataFrame so we can subset by index safely
    X_df = pd.DataFrame(X_all, index=ids_all)  # inde
    x_test = X_df.loc[X_want_test].values  # (N_test, D)

    # Geenrate one-hot-encoding vectors for each seq
    amino_acids = ["R", "K", "D", "E", "H", "N", "Q", "S", "T", "P", "C", "G", "A", "V", "I", "L", "M", "F", "Y", "W"]
    mapping = {aa: [int(i == j) for j in range(len(amino_acids))] for i, aa in enumerate(amino_acids)}

    def onehot_encode(seq, mapping):
        return np.array([mapping.get(i) for i in seq])

    x_test_one = np.array(pd.Series(X_want_test).apply(lambda seq: onehot_encode(seq, mapping)).tolist())

    return X_want_test, x_test,x_test_one


if __name__ == "__main__":
    import statistics
    # Parameters

    date = datetime.now().strftime('%Y%m%d')
    df_all_variants =pd.read_csv("./df_all_variant_0-2_mutations_long.csv")

    # Read and filter FASTA file
    fasta_path = "./file.txt"
    df = read_fasta_to_df(fasta_path)
    test_list=df['seq'].to_list()


    # Get model path from user
    model_name = input("What is the date of the model?\n"
                       "format:YYYYMMDD (in model_YYYYMMDD_100_best_parameters)\n")
    models_path = os.path.join("./Saved_models", f"model_{model_name}_100_best_parameters")

    if not os.path.exists(models_path):
        raise FileNotFoundError(f"Error: Model path {models_path} does not exist.")

    # Prepare results directory
    result_path = "./User_Predictions"
    os.makedirs(result_path, exist_ok=True)

    # Load models and predict
    model_groups = ['high','WT','SL','LO']
    models = load_models_from_paths(models_path, model_groups)
    # Generate input data
    X_want_test, x_test, x_test_one = generate_test_input(test_list)

    dfs = []
    for g in range(len(model_groups)):
        pred_list = []
        for i in range(10):
            y_pred = models[g][i].predict([x_test, x_test_one])
            y_pred_reshaped = y_pred.reshape(-1)
            pred_list.append(y_pred_reshaped)

        samples = X_want_test
        data = []
        gate_name = model_groups[g]
        for i in range(len(samples)):
            sample_data = {f"Prediction_{j + 1}": pred_list[j][i] for j in range(10)}
            sample_data["average predictions"] = statistics.mean(sample_data.values())
            data.append(sample_data)

        df_prediction = pd.DataFrame(data, index=samples)
        dfs.append(df_prediction)
        excel_file_path = os.path.join(result_path, f'predict_variants_{gate_name}.csv')
        df_prediction.to_csv(excel_file_path, index=True)

df_pred = pd.DataFrame(columns=['mutations', 'Er HI', 'Er WT', 'Er SL', 'Er LO', 'ddG pred','ddG norm to WT'], index=[])
predict_Hi, predict_WT, predict_SL, predict_LO = dfs[0], dfs[1], dfs[2], dfs[3]

###WT
preds_per_wt = []
wt_seq = 'TGPKARIVYGGR'
for d in dfs:
    if wt_seq in d.index:
        predict_p = d.loc[wt_seq, 'average predictions']
    else:
        predict_p = 0
    preds_per_wt.append(predict_p)

ER_hi_wt, ER_WT_wt, ER_SL_wt, ER_LO_wt = preds_per_wt
dG_wt = (-0.39661031 * ER_hi_wt - 0.3478708 * ER_SL_wt + 0.18383921 * ER_WT_wt + 0.59651285 * ER_LO_wt - 0.559550002114416)

for p in test_list:
    mut = df_all_variants[df_all_variants['mutated_sequence'] == p]['mutations'].values[0]

    if p in predict_Hi.index or p in predict_WT.index or p in predict_SL.index or p in predict_LO.index:

        preds_per_p = []
        real_per_p = []
        for d in dfs:
            if p in d.index:
                predict_p = d.loc[p, 'average predictions']
            else:
                predict_p = 0
            preds_per_p.append(predict_p)

        ER_hi_value, ER_WT_value, ER_SL_value, ER_LO_value = preds_per_p

        dG = (-0.39661031 * ER_hi_value - 0.3478708 * ER_SL_value + 0.18383921 * ER_WT_value + 0.59651285 * ER_LO_value - 0.559550002114416)
        ddG = dG - dG_wt
    df_pred.loc[p, :] = [mut, ER_hi_value, ER_WT_value, ER_SL_value, ER_LO_value, dG, ddG]

excel_file_path = os.path.join(result_path, f'{date}_predict_user_variants_combined_gates.csv')
df_pred.to_csv(excel_file_path, index=True)
print(f"Predictions saved in {os.path.abspath(excel_file_path)}")
