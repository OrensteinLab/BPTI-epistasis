#generate list of all posibilties:
import itertools
import pandas as pd

# Define the 12 positions
positions = [11,12,13,15,16,17,18,34,35,36,37,39] #for BPTI 1st project



# Define possible amino acid mutations for each position
# For simplicity, assuming each position can mutate to 'A', 'B', 'C'
amino_acids = ['A', 'R', 'N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

# Generate all unique combinations of positions (pairs)
position_pairs = list(itertools.combinations(positions, 2))

# Generate all possible double mutations for each pair of positions

sequence=list('TGPKARIVYGGR')
long_seq='RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRT'
position_to_index = {positions[i]: i for i in range(len(positions))}
data_path="./"
mutations = []
short_seq=[]
long=[]
X_all_pos_short_to_loc={}
number_of_mutations=[]
for pos1, pos2 in position_pairs:
    for mut1 in amino_acids:
        for mut2 in amino_acids:
            c=0
            mutation_description=[]
            seq_list = list(sequence)
            long_seq_list=list(long_seq)


            # Process first mutation
            if seq_list[position_to_index[pos1]] != mut1:
                mutation_description.append(f"{sequence[position_to_index[pos1]]}{pos1}{mut1}")
                c+=1
            seq_list[position_to_index[pos1]] = mut1
            long_seq_list[pos1-1]=mut1
            # Process second mutation
            if seq_list[position_to_index[pos2]] != mut2:
                mutation_description.append(f"{sequence[position_to_index[pos2]]}{pos2}{mut2}")
                c+=1
            seq_list[position_to_index[pos2]] = mut2
            long_seq_list[pos2 - 1] = mut2

            # Generate the mutated sequence and its description
            mutated_sequence = ''.join(seq_list)
            description_str = '_'.join(mutation_description)
            mutated_long_sequence = ''.join(long_seq_list)

            if mutated_sequence not in X_all_pos_short_to_loc:
                mutation_description = f"{sequence[position_to_index[pos1]]}{pos1}{mut1}_{sequence[position_to_index[pos2]]}{pos2}{mut2}"
                short_seq.append(mutated_sequence)
                long.append(mutated_long_sequence)
                mutations.append(description_str)
                X_all_pos_short_to_loc[mutated_sequence]=description_str
                number_of_mutations.append(c)
data={'mutations': mutations,'number of mutations': number_of_mutations,'mutated_sequence':short_seq,'long_mut_seq':long}
df_all_variant = pd.DataFrame(data)
X_all_pos=list(df_all_variant['mutated_sequence'])
df_all_variant.to_csv(f'{data_path}\df_all_variant_0-2_mutations_long.csv')
