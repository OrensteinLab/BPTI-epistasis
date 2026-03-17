import statistics
from collections import Counter
import numpy as np
import pandas as pd
import math
import keras
import tensorflow as tf
from keras.models import Sequential
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D,MaxPooling1D
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
start_time = time.time()


def reading_seq_from_file(file):
    """
    Reads sequence entries from a FASTQ-formatted file.

    :param file: str - path to the input FASTQ file
    :return: list - list of sequences extracted from the file
    """
    f = open(file, 'r')
    flag=0
    sequences=[]
    for line in f:
        if line.startswith('@'):
             flag=1

             continue
        elif flag==1:
            line = line.strip()
            sequences.append(line)
            flag=0
        else:
             continue
    f.close()
    return sequences


end_time = time.time()
print("Section 1 took:", end_time - start_time, "seconds")

start_time = time.time()
def sorting_dna_seq(seq_list,WT):
    """
    Filters DNA sequences by checking for the 'TAGC' primer, sufficient sequence length,
    and valid nucleotide content in the expected coding region.

    :param seq_list: list - list of DNA sequences to filter
    :param WT: str - wild-type amino acid sequence used to calculate the expected coding
        region length (3 nucleotides per amino acid)
    :return: list - DNA sequences that contain the 'TAGC' primer, are long enough after
        the primer, and do not contain characters other than A, T, C, and G in the
        inspected region
    """
    seq_list_after_sort = []
    missing_TAGC = 0
    short_seq = 0
    N_seq_count = 0
    short_seq_list=[]
    N_seq = []
    for seq in seq_list:

        start_primer = seq.find('TAGC')
        if start_primer == -1:
            missing_TAGC += 1
            continue
        if (len(seq) - (start_primer + 4)) < ((len(WT)) * 3):
            short_seq += 1
            short_seq_list.append(seq)
            continue
        if any(char not in {'A', 'T', 'C', 'G'} for char in
               seq[start_primer:start_primer + 4 + ((len(WT)) * 3)]) and start_primer != -1:

            #if seq.find('N', start_primer, (start_primer + 4 + ((len(WT) * 3))))!= -1:
            N_seq_count += 1
            N_seq.append(seq)
            continue
        #if seq.find('N', start_primer, (start_primer + 4 + ((len(WT)) * 3)) == -1) and start_primer != -1:
        seq_list_after_sort.append(seq)
    print('N in seq',N_seq_count)
    return seq_list_after_sort

def DNA_to_PROTEIN_LIST(seq_list,WT):
    """
    Translates filtered DNA sequences into protein sequences and summarizes mutation information.

    :param seq_list: list - list of DNA sequences to translate
    :param WT: str - wild-type amino acid sequence used as the reference for sequence length
        and allowed mutation positions
    :return: tuple - contains:
        protein_list (list): translated protein sequences of the expected length,
        mut_per_seq (dict): number of allowed mutations in each protein sequence,
        mut_not_in_place (int): number of sequences containing mutations outside the
            allowed positions,
        stop_count (int): number of sequences containing an internal stop codon,
        mut_acids_and_loc (dict): mutation annotation for each protein sequence,
        long_to_short_seq (dict): reduced sequence representation containing only the
            variable positions
    """

    protein_list = []
    CODON_TABLE = {
        'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
        'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
        'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
        'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
        'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
        'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
        'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
        'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }
    mut_per_seq = {}
    stop_count = 0
    mut_not_in_place = 0
    long_to_short_seq={}
    mut_acids_and_loc={}
    for s in seq_list:

        start_primer=s.find('TAGC')
        if start_primer != -1 and (len(s)-(start_primer+4))>=((len(WT))*3):
            line = s[(start_primer+4):(start_primer+4+3*(len(WT)))]
            seq_codons=[line[i:i + 3] for i in range(0, len(line), 3)]
            if 'TAA' in seq_codons[:54] or 'TGA' in seq_codons[:54] or 'TAG' in seq_codons[:54] :
                stop_count+=1
                continue

            aa_list=[]
            seq_mut_not_in_place=[]
            mut_count=0
            mut_acid=[]
            for i in range(start_primer+4,start_primer+4+(len(WT))*3,3):
                codon_T=s[i:i+3]
                codon_U=codon_T.replace('T','U')
                amino_acid= CODON_TABLE[codon_U]

                protein_loc = int(((i-start_primer) / 3))

                if WT[protein_loc-1]!=amino_acid and protein_loc not in [11,12,13,15,16,17,18,34,35,36,37,39]:
                    mut_not_in_place+=1
                    seq_mut_not_in_place.append(s)
                    break
                else:
                    aa_list.append(amino_acid)
                    if WT[protein_loc - 1] != amino_acid and protein_loc in [11,12,13,15,16,17,18,34,35,36,37,39]:
                        mut_count += 1
                        change=str(WT[protein_loc-1])+str(protein_loc)+amino_acid
                        mut_acid.append(change)
            separator='_'
            protein_seq=''.join(aa_list)
            all_protein_change=separator.join(mut_acid)
            if len(protein_seq)==(len(WT)):
                protein_list.append(protein_seq)
                short_seq = (protein_seq[10:13] + protein_seq[14:18] + protein_seq[33:37] + protein_seq[38])

                if protein_seq not in mut_per_seq:
                    long_to_short_seq[protein_seq]=short_seq
                    mut_per_seq[protein_seq] = mut_count
                    mut_acids_and_loc[protein_seq]=all_protein_change

        else:
            continue

    return protein_list, mut_per_seq,mut_not_in_place, stop_count,mut_acids_and_loc,long_to_short_seq

WT_pro_seq='RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRT'


def sorting_seq(seq_file_pre,WT_pro_seq):
    """
    Filters DNA sequences, translates them into protein variants, and counts the
    occurrence of each variant.

    :param seq_file_pre: list - list of raw DNA sequences to process
    :param WT_pro_seq: str - wild-type protein sequence used for DNA filtering and
        protein translation
    :return: tuple - contains:
        all_proteins (list): translated protein sequences,
        mut (dict): number of allowed mutations for each protein sequence,
        mut_acids_and_loc (dict): mutation annotation for each protein sequence,
        long_to_short_seq (dict): shortened representation of each protein sequence
            based on the variable positions,
        variant_count (Counter): number of occurrences of each protein variant
    """
    sorted_dna_seq=sorting_dna_seq(seq_file_pre,WT_pro_seq)
    all_proteins, mut,m_not_in_place,stop_pre,mut_acids_and_loc,long_to_short_seq = DNA_to_PROTEIN_LIST(sorted_dna_seq,WT_pro_seq)
    variant_count =Counter(all_proteins)

    return all_proteins, mut,mut_acids_and_loc,long_to_short_seq,variant_count

end_time = time.time()


print("Section 2 took:", end_time - start_time, "seconds")

start_time = time.time()
def summary_seq_data_raw(long_to_short_seq,mut_acids_and_loc, mut_per_seq, count_pre,count_hi,count_WT,count_SL,count_LO,):
    """
    Creates a summary DataFrame of variant identity, mutation annotation, and raw
    counts across the pre-sort library and all selection gates.

    :param long_to_short_seq: dict - mapping from full protein sequence to shortened
        sequence representation
    :param mut_acids_and_loc: dict - mutation annotation for each protein sequence
    :param mut_per_seq: dict - number of allowed mutations for each protein sequence
    :param count_pre: dict or Counter - counts of each variant in the pre-sort library
    :param count_hi: dict or Counter - counts of each variant in the high-affinity gate
    :param count_WT: dict or Counter - counts of each variant in the WT-affinity gate
    :param count_SL: dict or Counter - counts of each variant in the slightly lower-affinity gate
    :param count_LO: dict or Counter - counts of each variant in the low-affinity gate
    :return: pd.DataFrame - summary DataFrame indexed by shortened sequence representation,
        containing the full sequence, mutation annotation, number of mutations, and raw
        counts in the pre-sort library and each gate
    """
    df=pd.DataFrame(columns=['long seq','mutations', 'number of mutations','count pre','count high','count WT','count SL','count LO'],index=[])
    for v in count_pre:
        count_hi_value=count_hi.get(v,0)
        count_WT_value = count_WT.get(v, 0)
        count_SL_value = count_SL.get(v, 0)
        count_LO_value = count_LO.get(v, 0)
        df.loc[long_to_short_seq[v], :] = [v,mut_acids_and_loc[v], mut_per_seq[v], count_pre[v],count_hi_value,count_WT_value,count_SL_value,count_LO_value]
    return df



if __name__ == '__main__':
    import os
    import pandas as pd
    #from functions import fastq_reader, sort_seqs_to_mut, sort_mut_by_number, compare_seq, is_interface_mut


    data_path = os.path.join("./raw_data_names/")
    file_path = os.path.join("./pre_process_data_new")
    os.makedirs(file_path, exist_ok=True)

    file_pre = os.path.join(data_path,'Presorted.combined.extendedFrags.fastq')
    file_chymo_HI =os.path.join(data_path,'Chymotrypsin.HI.combined.extendedFrags.fastq')
    file_chymo_WT = os.path.join(data_path,'Chymotrypsin.WT.combined.extendedFrags.fastq')
    file_chymo_SL = os.path.join(data_path,'Chymotrypsin.SL.combined.extendedFrags.fastq')
    file_chymo_LO = os.path.join(data_path,'Chymotrypsin.LO.combined.extendedFrags.fastq')

    seq_file_pre = reading_seq_from_file(file_pre)
    seq_file_chymo_HI = reading_seq_from_file(file_chymo_HI)
    seq_file_chymo_WT = reading_seq_from_file(file_chymo_WT)
    seq_file_chymo_SL = reading_seq_from_file(file_chymo_SL)
    seq_file_chymo_LO = reading_seq_from_file(file_chymo_LO)


    df=pd.DataFrame()

    all_proteins_pre, mut_pre,mut_acids_and_loc_pre,long_to_short_seq,variant_count_pre=sorting_seq(seq_file_pre,WT_pro_seq)
    all_proteins_Hi, mut_Hi,mut_acids_and_loc_Hi,long_to_short_seq_Hi,variant_count_Hi=sorting_seq(seq_file_chymo_HI,WT_pro_seq)
    all_proteins_WT, mut_WT,mut_acids_and_loc_WT,long_to_short_seq_WT,variant_count_WT=sorting_seq(seq_file_chymo_WT,WT_pro_seq)
    all_proteins_SL, mut_SL,mut_acids_and_loc_SL,long_to_short_seq_SL,variant_count_SL=sorting_seq(seq_file_chymo_SL,WT_pro_seq)
    all_proteins_Lo, mut_Lo,mut_acids_and_loc_Lo,long_to_short_seq_LO,variant_count_LO=sorting_seq(seq_file_chymo_LO,WT_pro_seq)

    df_summary_raw=summary_seq_data_raw(long_to_short_seq,mut_acids_and_loc_pre,mut_pre,variant_count_pre,variant_count_Hi,variant_count_WT,variant_count_SL,variant_count_LO)
    df_summary_raw.to_csv(f'{data_path}\df_summary_raw_new.csv')




