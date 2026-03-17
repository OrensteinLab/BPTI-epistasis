import itertools
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from setuptools.command.rotate import rotate
from tensorflow import double

import pandas as pd
from functools import reduce

# =========================
# User-defined parameters
# =========================

date = datetime.now().strftime('%Y%m%d')

name = f"{date}_100_best_model"
#name = f"20260311_100_best_model"

data_path = f"./Predict_all_{name}"
hi_file=os.path.join(data_path,'predict_all_high.csv')
wt_file = os.path.join(data_path,'predict_all_WT.csv')
sl_file = os.path.join(data_path,'predict_all_SL.csv')
lo_file = os.path.join(data_path,'predict_all_LO.csv')
output_file = os.path.join(data_path,"combined_seed_predictions.xlsx")

# Name of the shared identifier column
# id_col = "variant"

# Prediction columns in each file are assumed to be like:
# prediction_1, prediction_2, ..., prediction_10
pred_prefix = "Prediction_"
n_seeds = 10

# =========================
# Read files
# =========================

df_hi = pd.read_csv(hi_file, index_col=0).reset_index()
df_wt = pd.read_csv(wt_file, index_col=0).reset_index()
df_sl = pd.read_csv(sl_file, index_col=0).reset_index()
df_lo = pd.read_csv(lo_file, index_col=0).reset_index()

df_hi = df_hi.rename(columns={"index": "variant"})
df_wt = df_wt.rename(columns={"index": "variant"})
df_sl = df_sl.rename(columns={"index": "variant"})
df_lo = df_lo.rename(columns={"index": "variant"})

id_col = "variant"
print(df_hi.columns.tolist())
# =========================
# Keep only ID + prediction columns
# =========================

pred_cols = [f"{pred_prefix}{i}" for i in range(1, n_seeds + 1)]

df_hi = df_hi[[id_col] + pred_cols].copy()
df_wt = df_wt[[id_col] + pred_cols].copy()
df_sl = df_sl[[id_col] + pred_cols].copy()
df_lo = df_lo[[id_col] + pred_cols].copy()

# =========================
# Rename columns by gate
# =========================

df_hi = df_hi.rename(columns={col: f"{col}_HI" for col in pred_cols})
df_wt = df_wt.rename(columns={col: f"{col}_WT" for col in pred_cols})
df_sl = df_sl.rename(columns={col: f"{col}_SL" for col in pred_cols})
df_lo = df_lo.rename(columns={col: f"{col}_LO" for col in pred_cols})

# =========================
# Merge all files on ID column
# =========================

dfs = [df_hi, df_wt, df_sl, df_lo]
df_merged = reduce(lambda left, right: pd.merge(left, right, on=id_col, how="inner"), dfs)

# =========================
# Reorder columns:
# variant, prediction_1_HI, prediction_1_WT, prediction_1_SL, prediction_1_LO, ...
# =========================

ordered_cols = [id_col]

for i in range(1, n_seeds + 1):
    ordered_cols.extend([
        f"{pred_prefix}{i}_HI",
        f"{pred_prefix}{i}_WT",
        f"{pred_prefix}{i}_SL",
        f"{pred_prefix}{i}_LO",
    ])

df_merged = df_merged[ordered_cols]

# =========================
# calculate one dG per seed
# dG = (-0.39661031 * HI) - (0.3478708 * SL) + (0.18383921 * WT) + (0.59651285 * LO) - 0.559550002114416
# =========================
for i in range(1, 11):
    df_merged[f"dG_{i}"] = (
        -0.39661031 * df_merged[f"Prediction_{i}_HI"]
        -0.3478708  * df_merged[f"Prediction_{i}_SL"]
        +0.18383921 * df_merged[f"Prediction_{i}_WT"]
        +0.59651285 * df_merged[f"Prediction_{i}_LO"]
        -0.559550002114416
    )

# =========================
# calculate mean and std of dG across seeds
# =========================
dg_cols = [f"dG_{i}" for i in range(1, 11)]

df_merged["dG_mean"] = df_merged[dg_cols].mean(axis=1)
df_merged["dG_std"] = df_merged[dg_cols].std(axis=1)

# =========================
# save
# =========================
df_merged.to_csv(output_file, index=False)

print(df_merged.head())
print(f"Saved to: {output_file}")
# =========================
# Save to Excel
# =========================

df_merged.to_excel(output_file, index=False)

print(f"Saved combined file to: {output_file}")

positions = [11, 12, 13, 15, 16, 17, 18, 34, 35, 36, 37, 39]
WT = 'TGPKARIVYGGR'
mutations = ['K', 'R', 'H',  # Positively Charged
             'D', 'E',  # Negatively Charged
             'G', 'C', 'S', 'T', 'N', 'Q', 'Y',  # Polar
             'F', 'W','A', 'V', 'I', 'L', 'M', 'P']  # Hydrophobic
position_to_index = {positions[i]: i for i in range(len(positions))}

df_all = df_merged.set_index('variant')

#generate dataframe for heatmap, if there is data from real pred values it add the value if not it
# add 'nan' then, genrate data for the heat map of dG
def data_for_heatmap(df_all,positions,mutations,position_to_index,value,WT,dg_col):

    """
    Generates a heatmap matrix of dG or ddG values for pairwise amino acid substitutions.

    :param df_all: DataFrame - DataFrame containing variant sequences and their associated values
    :param positions: list - list of mutable positions to include in the heatmap
    :param mutations: list - list of amino acid substitutions to evaluate
    :param position_to_index: dict - mapping from protein position to sequence index
    :param value: str - value type to calculate ('dG' or 'ddG')
    :param WT: str - wild-type protein sequence
    :param dg_col: str - name of the column containing dG values
    :return: DataFrame - heatmap matrix indexed by position and mutation pairs
    """
    position_pairs = list(itertools.combinations(positions, 2))

    data = pd.DataFrame(index=pd.MultiIndex.from_product([positions, mutations], names=['Position1', 'Mutation1']),
                           columns=pd.MultiIndex.from_product([positions, mutations], names=['Position2', 'Mutation2']))
    double_mutations = []
    X_all_pos_loc_to_short={}
    for pos1 in positions:
      for pos2 in positions:
        for mut1 in mutations:
            for mut2 in mutations:
                seq_list = list(WT)
                seq_list[position_to_index[pos1]] = mut1
                seq_list[position_to_index[pos2]] = mut2
                mutated_sequence = ''.join(seq_list)
                if pos1==pos2 or mutated_sequence not in df_all.index:
                  data.loc[(pos1, mut1), (pos2, mut2)] = np.nan

                else:
                  if value== 'dG':
                      data.loc[(pos1, mut1), (pos2, mut2)] = df_all.loc[mutated_sequence,dg_col]
                  if value== 'ddG':
                      ddG = df_all.loc[mutated_sequence, dg_col] - df_all.loc[WT, dg_col]
                      data.loc[(pos1, mut1), (pos2, mut2)] = ddG
    data_cleaned=data.apply(pd.to_numeric,errors='coerce')
    return data_cleaned



def heatmap(data,vmin,vmax,size,positions=None, value='ddG', mode='full', posix=None, posiy=None):
    """
    Plots a heatmap of predicted dG-, ddG-, or Gi-related values for variant combinations.

    :param data: DataFrame - matrix of values to display in the heatmap
    :param vmin: float - minimum value for the color scale
    :param vmax: float - maximum value for the color scale
    :param size: tuple - figure size
    :param positions: list or None - list of positions to label on the axes
    :param value: str - value type to display in the heatmap
    :param mode: str - display mode of the heatmap ('full' or 'avg')
    :param posix: int or None - specific x-position to display
    :param posiy: int or None - specific y-position to display
    :return: None
    """
    fig, ax = plt.subplots(figsize=size)
    if posix!=None and posiy !=None:
        specific_data=data.loc[posiy,posix]
    elif posix== None and posiy !=None:
        if value=='Gi':
            specific_data = data.loc[posiy, :].values.reshape(19, 228)
        else:
            specific_data = data.loc[posiy, :].values.reshape(20, 240)

    else:
        specific_data=data
    # Choose colormap
    if value == 'Gi':

        # cmap = LinearSegmentedColormap.from_list("BlueGreenRed", colors, N=256)
        colors_alt = [
            (1.0, 100 / 255, 0.0),  # deeper orange (strong negative)
            (1.0, 165 / 255, 0.0),  # orange (mild negative)
            (1.0, 1.0, 1.0),  # white (zero)
            (178 / 255, 102 / 255, 255 / 255),  # light purple (mild positive)
            (102 / 255, 0 / 255, 153 / 255)  # dark purple (strong positive)
         ]

        cmap = LinearSegmentedColormap.from_list("GreenGreyYellow", colors_alt, N=256)
    else:
        cmap = sns.color_palette("coolwarm", as_cmap=True)

    cmap_with_nan = cmap.copy()
    cmap_with_nan.set_bad("grey")

    # Plot the heatmap

    sns.heatmap(specific_data, cmap=cmap_with_nan, cbar=True, center=0,vmin=vmin, vmax=vmax, linewidths=0 if mode == 'avg' else 0, ax=ax)
    # Set colorbar title
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Predicted dGi', rotation=270, labelpad=15, fontsize=16)

    #cbar_kws={'shrink': 0.2}
    if mode == 'full':
        block_size = 19 if value == 'Gi' else 20
        tick_positions = np.arange(9.5, 228, 19) if value == 'Gi' else np.arange(10, 240, 20)
        grid_size = 12

        aa_colors = {'K': '#C71585', 'R': '#C71585', 'H' : '#C71585',  # Positively Charged
                     'D': '#000000', 'E':'#000000',  # Negatively Charged
                     'G': '#008000', 'C':'#008000', 'S': '#008000', 'T': '#008000', 'N': '#008000', 'Q':'#008000', 'Y':'#008000',  # Polar
                     'F': '#B8860B', 'W' :'#B8860B', 'A': '#B8860B', 'V': '#B8860B', 'I': '#B8860B', 'L': '#B8860B', 'M': '#B8860B', 'P': '#B8860B'}  # Hydrophobic

        for i in range(1, grid_size):
            ax.axvline(x=i * block_size, color='black', linewidth=0.5)
            ax.axhline(y=i * block_size, color='black', linewidth=0.5)


        # Labels
        if posix !=None and posiy !=None:
            ax.set_xlabel(f"Position {posix}")
            ax.set_ylabel(f"Position {posiy}")

            # Color each tick label
            for tick_label in ax.get_yticklabels():
                aa = tick_label.get_text()
                tick_label.set_color(aa_colors.get(aa, 'black'))
        elif posix == None and posiy != None:
            fil = data.loc[posiy, :]
            print('test', list(fil.columns))
            filtered_mutationsy = list(fil.index)
            col_labels = list(fil.columns)

            filt = [aa for pos, aa in col_labels]
            ax.set_yticklabels(filtered_mutationsy)
            ax.set_ylabel(f"Position {posiy}", fontsize=18)
            ax.set_xticks(tick_positions, positions)
            ax.set_xticks(np.arange(len(filt))+0.5)
            ax.set_xticklabels(filt,rotation=0, fontsize=10)


            # Color each tick label
            for tick_label in ax.get_yticklabels():
                aa = tick_label.get_text()
                tick_label.set_color(aa_colors.get(aa, 'black'))
            for tick_label in ax.get_xticklabels():
                aa = tick_label.get_text()
                tick_label.set_color(aa_colors.get(aa, 'black'))
        else:
            ax.set_xlabel("Position y", fontsize=16)
            ax.set_ylabel("Position x",fontsize=16)
            ax.set_xticks(tick_positions,positions, rotation=0, fontsize=12)
            ax.set_yticks(tick_positions, positions, rotation=90, fontsize=12)
    else:
        block_size = 19 if value == 'Gi' else 20
        tick_positions = np.arange(0.5, 12, 1) if value == 'Gi' else np.arange(0.5, 12, 1)
        grid_size = 12

        ax.set_xlabel("Position y", fontsize=16)
        ax.set_ylabel("Position x", fontsize=16)
        ax.set_xticks(tick_positions, positions, rotation=0, fontsize=12)
        ax.set_yticks(tick_positions, positions, rotation=90, fontsize=12)

    # Frame styling
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_aspect(1)
    ax.set_title(f'Predicted dGi of all single/souble mutatnt variants')
    plt.tight_layout()
    figuer_name=f'{value}_{mode}_{posiy}_{posix}_colors'
    #fig.savefig(fr"C:\python\BPTI NGS\figuers\new figures\{figuer_name}.tif", dpi=300, format="tiff", bbox_inches="tight")

    plt.show()


def avg_data(data, WT):
    """
    Calculates the average value for each position pair from a mutation-level data matrix.

    :param data: DataFrame - matrix containing values for all mutation combinations
    :param WT: str - wild-type sequence
    :return: DataFrame - position-by-position matrix of average values
    """
    ## average ddG
    data_avg = pd.DataFrame(index=pd.MultiIndex.from_product([positions], names=['Position1']),
                                columns=pd.MultiIndex.from_product([positions], names=['Position2']))

    double_mutations = []

    X_all_pos_loc_to_short = {}
    not_in_all = []
    same_pos = []
    for pos1 in positions:
        for pos2 in positions:
            block = data.loc[pos1, pos2]
            average = block.values.mean()
            data_avg.loc[pos1, pos2] = average
    data_avg = data_avg.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, set non-convertible values to NaN
    return data_avg


def epistasis(data_ddG,position_to_index, WT):
    """
    Calculates pairwise epistasis values from a ddG matrix for all non-wild-type mutation pairs.

    :param data_ddG: DataFrame - matrix of ddG values for single and double mutations
    :param position_to_index: dict - mapping from protein position to sequence index
    :param WT: str - wild-type protein sequence
    :return: DataFrame - matrix of epistasis values for pairwise mutation combinations
    """
    data_Gi = pd.DataFrame(index=pd.MultiIndex.from_product([positions, mutations], names=['Position1', 'Mutation1']),
                           columns=pd.MultiIndex.from_product([positions, mutations], names=['Position2', 'Mutation2']))
    data_Gi_cat = pd.DataFrame(index=pd.MultiIndex.from_product([positions, mutations], names=['Position1', 'Mutation1']),
                               columns=pd.MultiIndex.from_product([positions, mutations], names=['Position2', 'Mutation2']))
    seq_list=list(WT)

    nan_list = []
    zero = []
    double_mutations = []
    X_all_pos_loc_to_short = {}
    for pos1 in positions:
        for pos2 in positions:

            # Get WT amino acids at both positions
            pos1w = seq_list[position_to_index[pos1]]
            pos2w = seq_list[position_to_index[pos2]]

            # Filter out WT from mutation lists
            filtered_mutations1 = [mut for mut in mutations if mut != pos1w]
            filtered_mutations2 = [mut for mut in mutations if mut != pos2w]
            for mut1 in filtered_mutations1:
                for mut2 in filtered_mutations2:
                    pos1w = seq_list[position_to_index[pos1]]
                    pos2w = seq_list[position_to_index[pos2]]

                    Gd = data_ddG.loc[(pos1, mut1), (pos2, mut2)]
                    Gs1 = data_ddG.loc[(pos1, mut1), (pos2, pos2w)]
                    Gs2 = data_ddG.loc[(pos1, pos1w), (pos2, mut2)]
                    Gi = Gs1 + Gs2 - Gd

                    if pos1 == pos2:
                        data_Gi.loc[(pos1, mut1), (pos2, mut2)] = np.nan
                        data_Gi_cat.loc[(pos1, mut1), (pos2, mut2)] = np.nan

                    else:
                        data_Gi.loc[(pos1, mut1), (pos2, mut2)] = Gi


    # Identify valid rows/columns (excluding WT mutations)
    valid_rows = [(pos, mut) for pos in positions for mut in mutations if mut != seq_list[position_to_index[pos]]]
    valid_cols = valid_rows  # Same filtering for columns

    # Reindex DataFrame to exclude WT-containing rows and columns
    data_Gi = data_Gi.loc[valid_rows, valid_cols]
    data_Gi_cat = data_Gi_cat.loc[valid_rows, valid_cols]

    # Convert to numeric
    data_Gi_cleaned = data_Gi.apply(pd.to_numeric, errors='coerce')
    data_Gi_cat_cleaned = data_Gi_cat.apply(pd.to_numeric, errors='coerce')

    # Print new shape to verify
    print("New shape of data_Gi:", data_Gi.shape)  # Should be 228x228

    return data_Gi_cleaned


def epistasis_across_seeds(df_all, positions, mutations, position_to_index, WT, n_seeds=10):
    """
    Calculates ddG and epistasis matrices separately for each prediction seed.

    :param df_all: DataFrame - DataFrame containing variant sequences and seed-specific dG values
    :param positions: list - list of mutable positions
    :param mutations: list - list of amino acid substitutions
    :param position_to_index: dict - mapping from protein position to sequence index
    :param WT: str - wild-type protein sequence
    :param n_seeds: int - number of prediction seeds to process
    :return: tuple - dictionaries of ddG matrices and epistasis matrices for each seed
    """
    # NEW: dictionary to store ddG matrix of each seed
    ddg_by_seed = {}

    # NEW: dictionary to store epistasis matrix of each seed
    epistasis_by_seed = {}

    for i in range(1, n_seeds + 1):
        dg_col = f'dG_{i}'
        print(f'Processing {dg_col}...')

        # NEW: ddG for one specific seed
        data_ddG_seed = data_for_heatmap(
            df_all=df_all,
            positions=positions,
            mutations=mutations,
            position_to_index=position_to_index,
            value='ddG',
            WT=WT,
            dg_col=dg_col
        )

        # NEW: epistasis for the same specific seed
        data_Gi_seed = epistasis(
            data_ddG=data_ddG_seed,
            position_to_index=position_to_index,
            WT=WT
        )

        ddg_by_seed[dg_col] = data_ddG_seed
        epistasis_by_seed[dg_col] = data_Gi_seed

    return ddg_by_seed, epistasis_by_seed

def summarize_epistasis_across_seeds(epistasis_by_seed):
    """
    Calculates the mean and standard deviation of epistasis values across prediction seeds.

    :param epistasis_by_seed: dict - dictionary of epistasis matrices for different seeds
    :return: tuple - mean epistasis matrix and standard deviation epistasis matrix
    """
    seed_keys = list(epistasis_by_seed.keys())

    # NEW: stack all seed matrices into one 3D array
    matrices = np.stack([epistasis_by_seed[k].values for k in seed_keys], axis=0)

    # NEW: average across seed axis
    mean_matrix = np.nanmean(matrices, axis=0)

    # NEW: std across seed axis
    std_matrix = np.nanstd(matrices, axis=0)

    template = epistasis_by_seed[seed_keys[0]]

    epistasis_mean = pd.DataFrame(
        mean_matrix,
        index=template.index,
        columns=template.columns
    )

    epistasis_std = pd.DataFrame(
        std_matrix,
        index=template.index,
        columns=template.columns
    )

    return epistasis_mean, epistasis_std

def summarize_ddg_across_seeds(ddg_by_seed):
    """
    Calculates the mean and standard deviation of ddG values across prediction seeds.

    :param ddg_by_seed: dict - dictionary of ddG matrices for different seeds
    :return: tuple - mean ddG matrix and standard deviation ddG matrix
    """
    seed_keys = list(ddg_by_seed.keys())

    matrices = np.stack([ddg_by_seed[k].values for k in seed_keys], axis=0)

    mean_matrix = np.nanmean(matrices, axis=0)
    std_matrix = np.nanstd(matrices, axis=0)

    template = ddg_by_seed[seed_keys[0]]

    ddg_mean = pd.DataFrame(
        mean_matrix,
        index=template.index,
        columns=template.columns
    )

    ddg_std = pd.DataFrame(
        std_matrix,
        index=template.index,
        columns=template.columns
    )

    return ddg_mean, ddg_std

ddg_by_seed, epistasis_by_seed = epistasis_across_seeds(
    df_all=df_all,
    positions=positions,
    mutations=mutations,
    position_to_index=position_to_index,
    WT=WT,
    n_seeds=10
)
pos_test=[15,17,18]
ddg_mean, ddg_std = summarize_ddg_across_seeds(ddg_by_seed)
vmin=np.min(ddg_mean)
vmax=np.max(ddg_mean)
heatmap(ddg_mean,vmin,vmax,(20,7), positions,value='ddG', mode='full',posix=None,posiy=None)


avg_ddG=avg_data(ddg_mean, WT)
heatmap(avg_ddG,vmin,vmax,(20,7), positions, value='ddG', mode='avg',posix=None,posiy=None)
# for pos in positions:
#     heatmap(ddg_mean,vmin,vmax,(40,21),positions,value='ddG', mode='full',posix=None,posiy=pos)



epistasis_mean, epistasis_std = summarize_epistasis_across_seeds(epistasis_by_seed)
vmin_Gi=np.min(epistasis_mean)
vmax_Gi=np.max(epistasis_mean)
heatmap(epistasis_mean, vmin_Gi,vmax_Gi,(20,7),positions, value='Gi', mode='full',posix=None,posiy=None)

avg_Gi=avg_data(epistasis_mean, WT)
heatmap(avg_Gi,vmin_Gi,vmax_Gi,(10,8), positions, value='Gi', mode='avg',posix=None,posiy=None)

# for pos in positions:
#     heatmap(epistasis_mean,vmin_Gi,vmax_Gi,(40,21),positions,value='Gi', mode='full',posix=None,posiy=pos)

epistasis_output_path=os.path.join(data_path,"epistasis final.xlsx")
# with pd.ExcelWriter(epistasis_output_path) as writer:
#     epistasis_mean.to_excel(writer, sheet_name="mean")
#     epistasis_std.to_excel(writer, sheet_name="std")
#
# ddG_output_path=os.path.join(data_path,"ddG final check2.xlsx")
#
# with pd.ExcelWriter(ddG_output_path) as writer:
#     ddg_mean.to_excel(writer, sheet_name="mean")
#     ddg_std.to_excel(writer, sheet_name="std")