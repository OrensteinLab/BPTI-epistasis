#from sqlalchemy.dialects.mssql.information_schema import columns


def get_training_choice():
    """
    Prompt the user to choose a training methodology and validate the input.

    Available options:
    1 - Reserve 10% of the dataset as a test set and train on the remaining 90%.
    2 - Train on the full dataset without a separate test set.
    3 - Reserve 27 single-mutation variants as a test set and train on the remaining data.
    4 - Reserve 10 double-mutation variants as a test set and train on the remaining data.
    5 - Reserve variants mutated at a specific position as a test set and train on the remaining data.

    Returns:
        int: The selected training methodology (1, 2, 3, 4, or 5).
    """
    valid_choices = {"1", "2","3","4","5"}
    while (choice := input(
            "Select a training methodology:\n"
            "1: Use 10% of the data as a test set and train the model on the remaining 90%.\n"
            "2: Use the full dataset for training, without reserving a separate test set.\n"
            "3: Use 27 single mutation variants as a test set and train the model on the remaining data.\n"
            "4: Use 10 double mutation variants as a test set and train the model on the remaining data.\n"
            "5: Use variants mutated in a specific position as a test set and train the model on the remaining data.\n"

            "Enter your choice (1/2/3/4/5): "
    )) not in valid_choices:
        print("Invalid input. Please enter 1/2/3/4 or 5.")

    return int(choice)

def combine_gate_pred(dfs,df_all_variants,test_list):

    """
    Combines gate-specific predictions and calculates a predicted ddG value for each variant.

    :param dfs: list of pd.DataFrame - list of four prediction DataFrames ordered as
        [HI, WT, SL, LO], each indexed by mutated sequence and containing an
        'average predictions' column
    :param df_all_variants: pd.DataFrame - DataFrame containing all variants, including
        the columns 'mutated_sequence' and 'mutations'
    :param test_list: list - list of mutated sequences for which predictions will be combined
    :return: pd.DataFrame - DataFrame containing the mutation name, predicted ER values
        for the HI, WT, SL, and LO gates, and the calculated 'ddG pred' value
    """
    df_pred = pd.DataFrame(columns=['mutations', 'Er HI', 'Er WT', 'Er SL', 'Er LO', 'ddG pred'], index=[])
    predict_Hi, predict_WT, predict_SL, predict_LO = dfs[0],dfs[1],dfs[2],dfs[3]

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

            df_pred.loc[p, :] = [mut, ER_hi_value, ER_WT_value, ER_SL_value, ER_LO_value, dG]


    return df_pred

def train_and_evaluate(choice,hyperparameters,path,gate,target_position=None):
    """
    Trains models for a specific gate, saves the trained models, and optionally
    evaluates predictions according to the selected training scheme.

    :param choice: int - training scheme selector
        (1: 10% held-out test set,
         2: full-dataset training without a separate test set,
         3: 27 single-mutation variants as test set,
         4: 10 double-mutation variants as test set,
         5: position-wise held-out test set)
    :param hyperparameters: dict - dictionary containing the model hyperparameters,
        including architecture, dropout, learning rate, batch size, and number of epochs
    :param path: str - directory path for saving the trained models and prediction summaries
    :param gate: str - name of the target gate
    :param target_position: str or None - mutated position used for position-wise evaluation
        when choice == 5; otherwise None
    :return: pd.DataFrame or None - prediction summary DataFrame for choices 1, 3, 4,
        and 5; None for choice 2, where models are trained on the full dataset without
        evaluation on a separate test set
    """

    import keras
    import statistics
    X_want_test,x_train,x_train_one, x_test,x_test_one, y_train, y_test= prepare_data_for_model(enrich_gate, df_count_cut_gate, gate, X_all_pos,choice,target_position)
    parm = hyperparameters
    input_shape = (x_train.shape[1],)  # (D,)

    # Generating 10 predictions for each val set seq
    pred_list = []
    for seed in range(10):
        train_x_shuff, train_y_shuff, train_one_shuff = shuffle_data(x_train, y_train, x_train_one,seed)

        model = create_model(parm['arch'], input_shape, parm['DO'])
        opt = keras.optimizers.Adam(learning_rate=parm['learning_rate'])
        model.compile(loss='MSE', optimizer=opt, metrics=['mse'])

        model.fit([train_x_shuff, train_one_shuff], train_y_shuff, batch_size=parm['batch_size'],
                  epochs=parm['epoch'], verbose=0,  # sample_weight=normalized,
                  shuffle=True)

        # Save the model
        model_dir = os.path.join(path, gate)
        os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists
        if choice ==5:
            model_path = os.path.join(model_dir, f"model_{target_position}_{gate}_{seed}_{now}.h5")
        else:
            model_path = os.path.join(model_dir, f"model_{gate}_{seed}_{now}.h5")
        model.save(model_path)
        print(f"Trained {gate}, iteration {seed + 1}\n"
              f"Model saved at: {model_path}")
        if choice == 1 or choice == 3 or choice ==4 or choice == 5:
            y_pred = model.predict([x_test,x_test_one])
            y_pred_reshaped = y_pred.reshape(-1)
            pred_list.append(y_pred_reshaped)

    if choice == 1 or choice == 3 or choice ==4 or choice == 5:
        samples = X_want_test
        data = []

        for i in range(len(samples)):
            sample_data = {f"Prediction_{j + 1}": pred_list[j][i] for j in range(10)}
            sample_data["average predictions"] = statistics.mean(sample_data.values())
            if samples[i] in enrich_gate:
                sample_data["Actual_Value"] = enrich_gate[samples[i]]
            data.append(sample_data)
        df_prediction = pd.DataFrame(data, index=samples)

    if choice == 1:
        # Ensure the directory exists before saving
        test_summary_dir = os.path.join(PATH, "10_test_summary")
        os.makedirs(test_summary_dir, exist_ok=True)
        R=pearson_cal(df_prediction['average predictions'],df_prediction['Actual_Value'])
        print(f"Pearson correlaion of gate {gate}:",R)
        excel_file_path = os.path.join(test_summary_dir, f'10%_{gate}_final_data.csv')
        df_prediction.to_csv(excel_file_path, index=True)
        return df_prediction

    elif choice == 3:
        # Ensure the directory exists before saving
        test_summary_dir = os.path.join(PATH, "27_single_test_summary")
        os.makedirs(test_summary_dir, exist_ok=True)
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
        ddG=test_dic.values()
        df_prediction['EXP ddG']=ddG
        R=pearson_cal(df_prediction['average predictions'],df_prediction['EXP ddG'])
        print(f"Pearson correlaion of gate {gate}:",R)
        excel_file_path = os.path.join(test_summary_dir, f'27_single_{gate}_final_data.csv')
        df_prediction.to_csv(excel_file_path, index=True)
        return df_prediction

    elif choice == 4:

        # Ensure the directory exists before saving
        test_summary_dir = os.path.join(PATH, "10_double_test_summary")
        os.makedirs(test_summary_dir, exist_ok=True)
        # 10 double points
        test_dic = {'TGPRGRIVYGGR': 0.83072472, 'TGPRSRIVYGGR': 1.501384323, 'TGPRRRIVYGGR': 4.045516184,
                    'TGPRVRIVYGGR': 6.002537461, 'TGPRLRIVYGGR': 4.921405335, 'TGPRWRIVYGGR': 6.63304497,
                    'TGPYARLVYGGR': -3.49201313,
                    'TGARARIVYGGR': -0.588160841, 'TGPRAAIVYGGR': -0.4195144, 'TGPRARIAYGGR': -0.537152591}
        ddG=test_dic.values()
        df_prediction['EXP ddG']=ddG
        R=pearson_cal(df_prediction['average predictions'],df_prediction['EXP ddG'])
        print(f"Pearson correlaion of gate {gate}:",R)
        excel_file_path = os.path.join(test_summary_dir, f'10_double_{gate}_final_data.csv')
        df_prediction.to_csv(excel_file_path, index=True)
        return df_prediction

    elif choice == 5:
        # Ensure the directory exists before saving
        test_summary_dir = os.path.join(PATH, f"position_wise_{target_position}_test_summary")
        os.makedirs(test_summary_dir, exist_ok=True)

        excel_file_path = os.path.join(test_summary_dir, f'position_wise_{target_position}_{gate}.csv')
        df_prediction.to_csv(excel_file_path, index=True)
        R=pearson_cal(df_prediction['average predictions'],df_prediction['Actual_Value'])
        print(f"Pearson correlaion of position {target_position} in gate {gate}:",R)
        return df_prediction


if __name__ == '__main__':
    import time
    import os
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from BPTI_paper_Functions import (sorting_by_cutoff,enrichment_ratio_cal,prepare_data_for_model,shuffle_data,create_model,pearson_cal)

    # Initiate parameters for training models
    script_start_time = time.time()

    now = datetime.now().strftime("%Y%m%d")
    choice = get_training_choice()
    positions=[11,12,13,15,16,17,18,34,35,36,37,39]
    # Model file path
    if choice == 1:
        file_id = f"{now}_90_best_parameters"
    elif choice == 2:
        file_id = f"{now}_100_best_parameters"
    elif choice == 3:
        file_id = f"{now}_27_single_best_parameters"
    elif choice == 4:
        file_id = f"{now}_10_double_best_parameters"
    elif choice == 5:
        file_id = f"{now}_position_wise_best_parameters"

    PATH = f"./Saved_models/model_{file_id}"
    os.makedirs(PATH, exist_ok=True)

    # Load data
    df_raw =pd.read_csv("./df_summary_raw.csv",index_col=0)
    # Read the Excel with all the mutant variants possible (including wt, single and double)
    df_all_variants =pd.read_csv("./df_all_variant_0-2_mutations_long.csv",index_col=0)
    X_all_pos = list(df_all_variants['mutated_sequence'])

    df_count_cut_pre = sorting_by_cutoff(df_raw, 'presort')
    all_proteins_pre = list(df_count_cut_pre.index)


    # Optimized hyperparameters for each model
    hyperparameters = {
        "high": {'arch': [64, 32, 32, 32], 'batch_size': 256, 'DO': 0.1, 'learning_rate': 0.001, 'epoch': 50},
        "WT": {'arch': [64, 32, 32], 'batch_size': 32, 'DO': 0.3, 'learning_rate': 0.001, 'epoch': 30},
        "SL": {'arch': [64, 64, 32, 32], 'batch_size': 32, 'DO': 0.1, 'learning_rate': 0.001, 'epoch': 20},
        "LO": {'arch': [64, 64, 32, 32, 32], 'batch_size': 32, 'DO': 0.2, 'learning_rate': 0.001, 'epoch': 30}

    }
    gates=['high', 'WT', 'SL', 'LO']
    WT_pro_seq = 'TGPKARIVYGGR'
    dfs=[]
    pred_per_posi = pd.DataFrame(columns=['high','WT','SL','LO'],index=positions)
    # Iterate over each gate and train models
    for gate in gates:
        # Filter the data of the specific gate, include only up to 2 mutations and sort according to pre+gate count
        df_count_cut_gate = sorting_by_cutoff(df_count_cut_pre, gate)
        all_proteins_gate = list(df_count_cut_gate.index)

        # Prepare the label data
        enrich_gate, X, Y, counter_after_enrich, df_count_cut_gate = enrichment_ratio_cal(df_count_cut_pre,
                                                                                          df_count_cut_gate, WT_pro_seq,
                                                                                          all_proteins_gate,
                                                                                          all_proteins_pre, gate)
        if choice == 5:

            for target_position in positions:
                df_prediction = train_and_evaluate(choice, hyperparameters[gate], PATH, gate, target_position)
                R = pearson_cal(df_prediction['average predictions'], df_prediction['Actual_Value'])
                pred_per_posi.loc[target_position,gate]=R
            excel_file_path = os.path.join(PATH, f'position_wise_combine.csv')
            pred_per_posi.to_csv(excel_file_path, index=True)
        else:
            # train and evaluate the model
            target_position=None
            df_prediction=train_and_evaluate(choice,hyperparameters[gate],PATH,gate,target_position)
            dfs.append(df_prediction)
    if choice == 3:
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
        ddG=test_dic.values()
        test_list=list(test_dic.keys())
        # Ensure the directory exists before saving
        test_summary_dir = os.path.join(PATH, "27_single_test_summary")
        os.makedirs(test_summary_dir, exist_ok=True)
        df_combine=combine_gate_pred(dfs,df_all_variants,test_list)
        df_combine['ddG pred'] = pd.to_numeric(df_combine['ddG pred'], errors='coerce').fillna(0)
        df_combine['EXP ddG']=ddG
        R=pearson_cal(df_combine['ddG pred'],df_combine['EXP ddG'])
        print(f"Pearson correlaion of 27 single mutation variants prediction and ddG EXP:",R)
        excel_file_path = os.path.join(test_summary_dir, f'27_single_combine.csv')
        df_combine.to_csv(excel_file_path, index=True)

    if choice == 4:
        # Ensure the directory exists before saving
        test_summary_dir = os.path.join(PATH, "10_double_test_summary")
        os.makedirs(test_summary_dir, exist_ok=True)

        test_dic = {'TGPRGRIVYGGR': 0.83072472, 'TGPRSRIVYGGR': 1.501384323, 'TGPRRRIVYGGR': 4.045516184,
                    'TGPRVRIVYGGR': 6.002537461, 'TGPRLRIVYGGR': 4.921405335, 'TGPRWRIVYGGR': 6.63304497,
                    'TGPYARLVYGGR': -3.49201313,
                    'TGARARIVYGGR': -0.588160841, 'TGPRAAIVYGGR': -0.4195144, 'TGPRARIAYGGR': -0.537152591}
        ddG = test_dic.values()
        test_list = list(test_dic.keys())

        df_combine = combine_gate_pred(dfs,df_all_variants,test_list)
        df_combine['EXP ddG']=ddG
        df_combine['ddG pred'] = pd.to_numeric(df_combine['ddG pred'], errors='coerce').fillna(0)
        R=pearson_cal(df_combine['ddG pred'],df_combine['EXP ddG'])
        print(f"Pearson correlaion of 10 double mutation variants combine gate prediction and ddG EXP:",R)
        excel_file_path = os.path.join(test_summary_dir, f'10_double_combine.csv')
        df_combine.to_csv(excel_file_path, index=True)

