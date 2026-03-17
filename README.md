# Machine-Learning Prediction of Affinity and Epistasis in the Bovine Pancreatic Trypsin Inhibitor–Chymotrypsin Complex

# Introduction

We present a machine-learning approach using deep neural networks trained on High-Throughput 
Sequencing (HTS) data from protein mutagenesis library affinity screens. In this study,
we focused on the interaction between BPTI and chymotrypsin, using experimental data from a 
combinatorial library with mutations at twelve binding-interface positions. 
Our goal is to accurately predict the binding effects of observed and unobserved variants,
reconstruct the affinity landscape, and analyze epistatic interactions between mutations.

# Purpose
This repository contains the code related to our research project,
which aims to reconstruct and analyze the binding affinity landscape of the BPTI–chymotrypsin
interaction. Using deep neural networks trained on deep mutational scanning data,
we predict binding free energy changes (ΔΔG<sub>bind</sub>) for protein variants and quantify
epistatic interactions between mutations. The framework enables the identification of sequence
patterns and mutation combinations that modulate binding affinity.

# Data
The experimental raw data used in this study consists of BPTI variants in complex with
chymotrypsin. All variants in this dataset contain mutations only at twelve interface positions:
11, 12, 13, 15, 16, 17, 18, 34, 35, 36, 37, and 39. High-throughput sequencing data from 
affinity-sorted populations was used to train deep neural networks, enabling accurate prediction
of binding free energy changes for observed and unobserved variants, as well as analysis of
epistatic interactions between mutations.

# Setup Environment Instructions (Windows)

Before you proceed with the setup, make sure to have Python and Anaconda installed on your system.


1. **Download the Code Repository:**
   - Visit the GitHub repository: [https://https://github.com/OrensteinLab/BPTI-epistasis](https://github.com/OrensteinLab/N-TIMP2--MMP-selectivity)
   - Download the contents of the "Code" folder.

2. **Inside the "Code" Folder, place the pre-process data:**

   For code validation, include the following datasets:

   - df_summary_raw.csv
   - df_all_variant_0-2_mutations_long.csv
   - X_esm_delta_mean_all_pos_640_par.npy
   - variant_ids_all_pos.npy

3. **Create a Virtual Conda Environment:**
   - Open a command prompt.
   - Navigate to the directory where you downloaded the "Code" repository.
   - Run the following command to create a virtual conda environment named "bpti_env" with Python 3.9.7 and the required modules:
     
     ```bash
     conda create --name bpti_env python=3.9.7 --file Requirements.txt 
     ```

4. **Activate the New Environment and Run the Script:**
   - Activate the environment using the following command:
     ```bash
     conda activate bpti_env
     ```
   - Run the scripts according to the provided usage instructions.


# Usage

### 1.	Pre-processing Script (Optional):
#### 1.1 Reading FASTA files
The script `Reading_ngs_file.py` processes FASTA files to count valid variants. A valid variant must:
 
  * Align with the BPTI wild-type amino acid sequence.
  * Contain mutations only at the relevant positions.

Output: 
  * The script merges data from four affinity gates sequencing FASTA files into csv file called "df_summary_raw".

#### 1.2 Generating all single- and double-mutation variants
The script `all_variants_options.py` generates a summary of all possible variants in the library, including the wild type, single mutants, and double mutants. For each variant, the script records the mutation description, the sequence of the 12 relevant positions, and the full-length protein sequence.

Output:
  * A csv file called "df_all_variant_0-2_mutations_long" that summarizing all variants, including their mutation notation, 12-position sequence, and full-length sequence.

#### 1.3 Generating ESM embeddings for the full-length protein and avg. across twelve position
The script `model_ESM_640_param.py` generates sequence embeddings for each full-length protein variant using the ESM-2 model. For each variant, the script extracts the embeddings of the 12 relevant positions, averages them, and saves the resulting 640-parameter representation.

Output:
  * A file called "X_esm_delta_mean_all_pos_640_par" containing the averaged embeddings for all variants (wild type, single mutants, and double mutants).
  * A file called "variant_ids_all_pos" containing the variant IDs together with their twelve-position mutated sequences


Important Note: These scripts is for users who wish to train a model based on their own data. To work with our data and predict BPTI variants, you don't need to run these scripts. Because the output files of these scripts (df_summary_raw.csv, df_all_variant_0-2_mutations_long.csv, X_esm_delta_mean_all_pos_640_par.npy, variant_ids_all_pos.npy) are already provided in the data folder.



### 2.	Train models script:
Run `train_model.py`. You will be prompted to choose an action:
1.	Use 10% of the data as a test set and train the model on the remaining 90%.
2.	Use the full dataset for training, without reserving a separate test set.
3.	Use 27 single-mutant variants as a test set and train the model on the remaining data.
4.	Use 10 double-mutant variants as a test set and train the model on the remaining data.
5.	Use variants mutated at a specific position as a test set and train the model on the remaining data.

Output:
* Trained models will be saved in sub folder in "Saved_models".
* If a test-set option is selected, prediction files will be generated at the end of execution.

### 3.	Get predictions:
After training the model by running: python `train_model.py` using option number 2, you can now generate predictions.
Each script will prompt you to enter the model date in YYYYMMDD format.

#### 3.1	Predicting All possibole Mutations:
The script `predict_all_library.py` generates predictions for all possible single and double mutations.

Output:
* The script creates a csv file 'Predict_all_{gate}, with the Predicted_log2_ER for each gate.
* The script creates a csv file summary predict all variants combined gates, with the Predicted_log2_ER of 4 affinity gates and ddG<sub>bind</sub> of each variant. 

#### 3.2	User-Specified Variant Predictions:
The script `predict_user_variants.py` predicts the log2 ER and final ddG<sub>bind</sub> of any variant.
To use this script, create a TXT file containing twelve amino acid protein sequences (of the twelve relevant positions). 
Enter the sequence in FASTA format:
```bash
>variant_K15I
TGPIARIVYGGR
```
Output:
* The script creates a csv file '{current data}_user_predictions', with the Predicted_log2_ER and ddG<sub>bind</sub> of each variant.

#### 3.3	Affinity and epistasis analysis::

The script `final_ddG_epistasis_and_heatmaps.py` processes the predicted binding affinities for the full mutational library (generated in the `predict_all_library.py` script) and performs downstream analysis of the BPTI–chymotrypsin binding landscape.
The script calculates the binding free energy changes (ΔΔG<sub>bind</sub>) and epistasis values for all variants based on the predicted enrichment ratios. It then summarizes the results and visualizes the affinity and epistasis landscapes
Output:
* Combined_seed_predictions saved as Excel file containing the predicted enrichment values for each variant from all ensemble seeds and all four selection gates (HI, WT, SL, LO), organized so that predictions from the same seed across gates appear together.
* ΔΔG<sub>bind</sub> and epistasis matrices saved as Excel files: one sheet for predicted mean values and one sheet for the corresponding standard deviations across ensemble models.
* Heatmaps of the affinity and epistasis landscapes, visualizing ΔΔG<sub>bind</sub> values and epistatic interactions across the mutation space.
* Position-pair heatmaps summarizing the average affinity and average epistasis for each pair of mutated positions.

During execution, you will be prompted to enter the date of the prediction folder in YYYYMMDD format.


