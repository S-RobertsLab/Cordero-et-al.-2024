from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm, chi2_contingency

def reverse_complement(seq: str):
    """returns the reverse complement of nucleotide sequences in standard or IUPAC notation

    Args:
        seq (str): sequence of DNA in standard or IUPAC form that

    Returns:
        reverse_complement (str): the reverse complement of the input sequence
    """
    # make a lookup table
    complement_table = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "R": "Y",
        "Y": "R",
        "S": "S",
        "W": "W",
        "K": "M",
        "M": "K",
        "B": "V",
        "D": "H",
        "H": "D",
        "V": "B",
        "N": "N"
    }

    seq_rev = seq[::-1]
    complement_seq = "".join(complement_table.get(base, base) for base in seq_rev)
    return complement_seq

def bedtools_intersect(reference_file, input_file):
    """Takes a reference file and an input file and generates an intersect file containing the overlapping regions.
    
    Args:
        reference_file (file): .bed file of the genes you want to compare with
        input_file (file): .bed file of the genes you want to compare with
    """
    reference_file = Path(reference_file)
    input_file = Path(input_file)
    intersect_file = input_file.with_name(f"{input_file.stem}_{reference_file.stem}.intersect")
    if intersect_file.exists():
        return intersect_file
    else:
        command = f"bedtools intersect -wa -wb -a {reference_file} -b {input_file} > {intersect_file}"
        subprocess.run(command, shell=True, check=True)

        return intersect_file

def mutation_combinations(mut_type: str):
    """Takes a mutation type in the format N>N and returns a list of all possible nucleotide mutations.

    Args:
        mut_type (str): the mutation type in the format N>N

    Returns:
        possible_mutations (list): a list of all possible nucleotide mutations
    """
    iupac_trans = {
        "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT",
        "M": "AC", "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG",
        "N": "ACGT", 'A': 'A', 'T': 'T', 'C': 'C', 'G': 'G'
    }
    
    # Split the mutation type at the '>' character
    from_nuc, to_nuc = mut_type.split('>')
    
    # Get the possible nucleotides for each side of the '>'
    from_nucs = iupac_trans[from_nuc]
    to_nucs = iupac_trans[to_nuc]
    
    # Generate all possible combinations excluding same nucleotides
    possible_mutations = [f"{f}>{t}" for f in from_nucs for t in to_nucs if f != t]
    
    return possible_mutations

def create_normalization_file(reference_file):
    reference_file = Path(reference_file)
    length = 0
    
    with open(reference_file, 'r') as f:
        for line in f:
            start, stop = line.strip().split('\t')[1:3]
            length += int(stop) - int(start)
    
    return length

def create_matrix(intersect_file):
    """Takes an intersect file and generates a matrix of counts for each mutation type.

    Args:
        intersect_file (file): .bed file of the genes you want to compare with
    """
    intersect_file = Path(intersect_file)
    ts_matrix_file = intersect_file.with_name(f"{intersect_file.stem}_TS_counts_matrix.tsv")

    # check if this has been done already
    if ts_matrix_file.exists():
        return ts_matrix_file

    else:
        # use a dictionary to store strandedness, mutation type, and context counts in that order
        txn_counter = {mutation: {'TS':0,'NTS':0} for mutation in mutation_combinations("N>N")}

        # Calculate the total number of lines in the file
        with open(intersect_file, 'r') as f:
            total_lines = sum(1 for _ in f)

        with open(intersect_file, 'r') as f:
            for line in tqdm(f, total=total_lines, unit='line', desc="Counting Mutations", ncols=100, mininterval=.1):
                line = line.strip().split('\t')
                mutation_type = line[-2]
                
                if mutation_type != 'SNV':
                    continue

                ref_strand = line[5]
                mutation = line[-3]
                mutation_strand = line[-5]

                if ref_strand == '+' == mutation_strand:
                    txn_counter[mutation]['NTS'] += 1
                elif ref_strand == '-' != mutation_strand:
                    txn_counter[mutation]['TS'] += 1

        # create a pandas df where the columns are the contexts and the rows are the counts
        ts_df = pd.DataFrame(txn_counter).T

        # Save the DataFrame to a file
        ts_df.to_csv(ts_matrix_file, sep='\t')

        return ts_matrix_file

def normalize_counts(ts_matrix, gene_ref_file):
    ts_matrix = Path(ts_matrix)
    genes_length = create_normalization_file(gene_ref_file)
    
    # Define output file names
    normalized_ts = ts_matrix.with_name(f"{ts_matrix.stem}_normalized.tsv")

    # Read the input data
    ts_df = pd.read_csv(ts_matrix, sep='\t', header=0, index_col=0)

    # Normalize the TS dataframe
    normalized_ts_df = ts_df.div((2*genes_length)/1e7)
    # Save the normalized TS dataframe to file
    normalized_ts_df.to_csv(normalized_ts, sep='\t')

    return normalized_ts

def combine_complements(normalized_ts, pyrimidine=True):
    """Takes the normalized transcribed and non-transcribed counts and combines the complementary mutations."""
    normalized_ts = Path(normalized_ts)
    combined_counts_file = normalized_ts.with_name(f"{normalized_ts.stem}_combined.tsv")

    # Read the input data
    normalized_ts_df = pd.read_csv(normalized_ts, sep='\t', header=0, index_col=0)

    # Define the complementary pairs
    complementary_pairs = {
        'A>C': 'T>G', 'T>G': 'A>C',
        'A>G': 'T>C', 'T>C': 'A>G',
        'A>T': 'T>A', 'T>A': 'A>T',
        'C>A': 'G>T', 'G>T': 'C>A',
        'C>G': 'G>C', 'G>C': 'C>G',
        'C>T': 'G>A', 'G>A': 'C>T'
    }

    # Get the desired mutations
    mutations = mutation_combinations('Y>N' if pyrimidine else 'R>N')

    # Combine the complementary mutations
    combined_counts = pd.DataFrame(index=mutations, columns=['TS', 'NTS'], data=0.0)

    for mutation in mutations:
        complement = complementary_pairs[mutation]
        combined_counts.loc[mutation, 'TS'] = normalized_ts_df.loc[mutation, 'TS'] + normalized_ts_df.loc[complement, 'NTS']
        combined_counts.loc[mutation, 'NTS'] = normalized_ts_df.loc[mutation, 'NTS'] + normalized_ts_df.loc[complement, 'TS']

    # Save the combined counts DataFrame to a file, if pyrimidine keep the order the same, if purine reverse the order of the rows
    combined_counts = combined_counts if pyrimidine else combined_counts[::-1]
    combined_counts.to_csv(combined_counts_file, sep='\t')

    return combined_counts_file

def set_nature_style():
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1

def plot_transcription_data(file_path):
    """
    Reads transcription data from a file and plots as bar plots and their log2 ratios.

    Parameters:
    file_path (str): Path to the file containing the transcription data.
    """
    # Read the input data
    data = pd.read_csv(file_path, sep='\t', index_col=0)

    # normalize the df so that its all normalized to the mean of the data
    # data = data / data.mean().mean()

    # Set the aesthetic style
    set_nature_style()

    # Ensure the mutations are the index
    data.index.name = 'Mutation'

    # Separate the data into transcribed and non-transcribed for clarity in plotting
    transcribed = data['TS']
    non_transcribed = data['NTS']

    # Calculate log2 ratios
    log2_ratios = np.log2(non_transcribed / transcribed)  # Handle potential division by zero or log of zero based on your data.

    # Create a figure and axes with specific layout settings
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})

    # Define the width of the bars and the positions
    width = 0.35  # the width of the bars
    ind = np.arange(len(transcribed))  # the x locations for the groups

    # Plotting the bars (transcribed and non_transcribed)
    p1 = ax.bar(ind - width/2, non_transcribed, width, color='IndianRed', label='Non-Transcribed', edgecolor='black', linewidth=1.5)
    p2 = ax.bar(ind + width/2, transcribed, width, color='SkyBlue', label='Transcribed', edgecolor='black', linewidth=1.5)

    # Customizing the ax (main bar plot)
    ax.set_xticks(ind)
    ax.set_xticklabels(transcribed.index, fontsize=12)  # Rotation for better readability
    ax.set_title('Transcribed Strand Asymmetry', fontsize=20, weight='bold')
    ax.set_ylabel('Mutations per Mb', fontsize=16, weight='bold')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.5, 
              facecolor='white', edgecolor='black', frameon=True, 
              borderpad=0.2, labelspacing=0.2, handletextpad=0.2)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Customizing the ax2 (log2 ratios)
    colors = ['SkyBlue' if ratio < 0 else 'IndianRed' for ratio in log2_ratios]  # Flipped colors based on the ratio
    ax2.bar(ind, log2_ratios, color=colors, edgecolor='black', linewidth=1)
    ax2.axhline(y=0, color='black', linewidth=1.5)  # Adding horizontal line at y=0
    ax2.set_xticks(ind)
    ax2.set_xticklabels(transcribed.index, fontsize=12)  # Consistency in labels with the main plot, rotation for readability
    ax2.set_ylabel('Log2 Ratio', fontsize=16, weight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_ylim([-1, 1])

    # Show the plot
    plt.show()

def main(reference_file, input_file, genome_file, pyrimidine=True):
    # Define the paths to the input files
    reference_file = Path(reference_file)
    input_file = Path(input_file)
    genome_file = Path(genome_file)

    # intersect the files and count
    intersect_file = bedtools_intersect(reference_file, input_file)
    ts_matrix = create_matrix(intersect_file)
    ts_norm = normalize_counts(ts_matrix, reference_file)
    combined_data = combine_complements(ts_norm, pyrimidine=pyrimidine)
    
    # plot the transcription data and do some stats
    plot_transcription_data(combined_data)

if __name__ == "__main__":
    main(
        reference_file='/home/cam/Documents/repos/AsymTools2/RPE-1_asym_files/ensGeneList_condensed_no_zero.bed',
        input_file='/media/cam/Storage/8-oxodG/REVISIONS/test_asym_old/polh_kbr.bed',
        genome_file='/home/cam/Documents/genome_files/hg19/hg19.fa',
        pyrimidine=False
    )