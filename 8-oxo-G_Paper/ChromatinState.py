from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import numpy as np
from scipy.stats import ttest_rel

def intersect_mutations_with_map(mutations, map_file):
    """Intersect mutations with a map file"""
    # Create Path objects
    mutations = Path(mutations)
    map_file = Path(map_file)
    # Create a temporary file
    output_file = Path(mutations).with_suffix(f'.{map_file.stem}.intersect')
    # Intersect the files
    if output_file.exists():
        return output_file
    cmd = f'bedtools intersect -a {mutations} -b {map_file} -wa -wb > {output_file}'
    subprocess.run(cmd, shell=True)
    # Return output file
    return output_file

def count_intersected_file(intersected_file):
    """Read the intersected file and count number of mutations intersecting each region per sample"""
    # Read the intersected file
    intersected_file = Path(intersected_file)
    if intersected_file.with_suffix('.counts').exists():
        return pd.read_csv(intersected_file.with_suffix('.counts'), sep='\t', index_col=0, header=0)
    with open(intersected_file) as f:
        counts = {}
        for line in f:
            line = line.strip().split()
            mutation_type, sample, state = line[8], line[9], line[13]
            if mutation_type != 'SNV': continue
            if sample not in counts:
                counts[sample] = {}
            if state not in counts[sample]:
                counts[sample][state] = 0
            counts[sample][state] += 1
    # Create a DataFrame
    df = pd.DataFrame(counts).T.fillna(0)
    # sort the columns in ascending order for the number before the first "_"
    df = df.reindex(sorted(df.columns, key=lambda x: int(x.split('_')[0])), axis=1)
    # reverse the column order
    df = df[df.columns[::-1]]
    # sort the index
    df = df.sort_index()
    # Write dataframe to tsv text file
    df.to_csv(intersected_file.with_suffix('.counts'), sep='\t')
    # Return DataFrame
    return df

def count_region_sizes_in_chrom_map(map_file):
    """Count the sizes of each region in the chromatin map"""
    # Read the map file
    map_file = Path(map_file)
    if map_file.with_suffix('.sizes').exists():
        return pd.read_csv(map_file.with_suffix('.sizes'), sep='\t', index_col=0, header=0)
    with open(map_file) as f:
        counts = {}
        for line in f:
            line = line.strip().split()
            state = line[3]
            if state not in counts:
                counts[state] = 0
            length = int(line[2]) - int(line[1])
            counts[state] += length
    # Create a DataFrame
    df = pd.DataFrame(counts, index=['size']).T
    # sort the index by the number before the first "_" in decreasing order
    df = df.reindex(sorted(df.index, key=lambda x: int(x.split('_')[0]), reverse=True))
    # Write dataframe to tsv text file
    df.to_csv(map_file.with_suffix('.sizes'), sep='\t')
    # Return DataFrame
    return df

def normalize_data_to_mutations_per_mb(intersect_counts_df, region_sizes_df):
    """Normalize the data to mutations per Mb"""
    # copy the intersected counts dataframe
    normalized_counts = intersect_counts_df.copy()
    # divide each count by the size of the region and multiply by 1e6
    for sample in normalized_counts.index:
        for state in normalized_counts.columns:
            normalized_counts.loc[sample, state] = normalized_counts.loc[sample, state] / region_sizes_df.loc[state, 'size'] * 1e6
    # return the normalized counts
    return normalized_counts

def plot_bargraphs(normalized_counts_df):
    """Plot bar graphs for different chromatin domains with T-shaped error bars and custom styling."""
    # Set Seaborn style but remove vertical gridlines and keep horizontal gridlines
    sns.set(style="whitegrid", rc={'axes.grid.axis': 'y', 'axes.grid': True})
    # override the colors of the axis lines to be black
    plt.rcParams['axes.edgecolor'] = 'black'

    # Define the chromatin state groups
    domains = {
        'Heterochromatin': ['13_Heterochrom/lo', '12_Repressed', '8_Insulator'],
        'Promoter': ['3_Poised_Promoter', '2_Weak_Promoter', '1_Active_Promoter'],
        'Enhancer': ['7_Weak_Enhancer', '6_Weak_Enhancer', '5_Strong_Enhancer', '4_Strong_Enhancer'],
        'Transcribed': ['11_Weak_Txn', '10_Txn_Elongation', '9_Txn_Transition']
    }

    fig, axes = plt.subplots(1, 4, figsize=(8, 4), sharey=True)
    fig.suptitle('Mutations Across Chromatin Domains', fontsize=16)

    bar_color = sns.color_palette("Blues")[2]  # Use a lighter blue from the palette
    # use a pink bar color RGB(238,42,123) 
    # bar_color = (0.9333333333333333, 0.16470588235294117, 0.4823529411764706)

    for i, (domain, states) in enumerate(domains.items()):
        mean = normalized_counts_df[states].mean()
        std = normalized_counts_df[states].std()

        # Convert underscores to spaces and remove leading numbers
        labels = [label.split('_', 1)[-1].replace('_', ' ') for label in mean.index]

        # Set the formatted labels and rotate them 45 degrees
        axes[i].set_xticklabels(labels, rotation=45, ha='right')

        # Plot the bar graph with T-shaped error bars
        axes[i].bar(mean.index, mean, yerr=std, color=bar_color, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})

        # Plot individual points (replicates) over the bars
        for j, state in enumerate(states):
            jitter = np.linspace(-0.1, 0.1, len(normalized_counts_df[state]))  # Add small jitter
            # axes[i].scatter(np.full_like(normalized_counts_df[state], j) + jitter, normalized_counts_df[state], color='b', alpha=0.7)

        axes[i].set_title(domain, fontsize=12)

        if i == 0:
            axes[i].set_ylabel('Lesions per Mb')
            axes[i].tick_params(axis='y', which='both', left=True, length=4)  # Re-enable tick marks on the y-axis and make them a little shorter

        # Re-enable tick marks on both axes
        axes[i].tick_params(axis='x', which='both', bottom=True, length=4)

        # set the ymin to 0
        axes[i].set_ylim(bottom=0)

        # Despine to remove axes borders
        sns.despine(ax=axes[i])

    # Adjust layout to reduce space between subplots
    plt.subplots_adjust(wspace=-1)  # Reduce spacing between plots

    # Overall layout adjustments to avoid overlap with title
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    ######################################################################
    normalized_counts_df.to_csv('/media/cam/Working/Data_Table/normalized_counts_df.tsv', sep='\t')






def compare_domains(normalized_counts_df):
    """Compare mutation counts between chromatin domains with paired t-tests and return p-values."""
    
    # Define the chromatin state groups
    domains = {
        'Heterochromatin': ['13_Heterochrom/lo', '12_Repressed', '8_Insulator'],
        'Promoter': ['3_Poised_Promoter', '2_Weak_Promoter', '1_Active_Promoter'],
        'Enhancer': ['7_Weak_Enhancer', '6_Weak_Enhancer', '5_Strong_Enhancer', '4_Strong_Enhancer'],
        'Transcribed': ['11_Weak_Txn', '10_Txn_Elongation', '9_Txn_Transition']
    }
    
    # Create a DataFrame to store the p-values for the comparisons
    comparison_results = pd.DataFrame(index=domains.keys(), columns=domains.keys())

    # Iterate through each pair of domains and perform the paired t-test
    for domain1, states1 in domains.items():
        for domain2, states2 in domains.items():
            if domain1 != domain2:
                # Get the mean values for each domain, paired by sample
                mean_domain1 = normalized_counts_df[states1].mean(axis=1)
                mean_domain2 = normalized_counts_df[states2].mean(axis=1)

                # Perform the paired t-test
                t_stat, p_value = ttest_rel(mean_domain1, mean_domain2)

                # Store the p-value in the comparison matrix
                comparison_results.loc[domain1, domain2] = p_value
            else:
                # Set p-value to NaN when comparing the same domain
                comparison_results.loc[domain1, domain2] = None

    # Perform a Bonferroni correction for multiple testing
    comparison_results = comparison_results * (len(comparison_results.columns)-1)

    return comparison_results

def compare_all_states(normalized_counts_df):
    """Compare mutation counts between individual chromatin states with paired t-tests and return p-values."""
    
    # Get the column names, which represent the chromatin states
    states = normalized_counts_df.columns
    # remove the states we dont care about
    states = [state for state in states if state.split('_')[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']]
    
    # Create a DataFrame to store the p-values for the comparisons
    state_comparison_results = pd.DataFrame(index=states, columns=states)

    # Iterate through each pair of chromatin states and perform the paired t-test
    for state1 in states:
        for state2 in states:
            if state1 != state2:
                # Get the data for each state, paired by sample
                data_state1 = normalized_counts_df[state1]
                data_state2 = normalized_counts_df[state2]

                # Perform the paired t-test
                t_stat, p_value = ttest_rel(data_state1, data_state2)

                # Store the p-value in the comparison matrix
                state_comparison_results.loc[state1, state2] = p_value
            else:
                # Set p-value to NaN when comparing the same state
                state_comparison_results.loc[state1, state2] = None

    # Perform a Bonferroni correction for multiple testing
    state_comparison_results = state_comparison_results * (len(state_comparison_results.columns)-1)

    return state_comparison_results

def main(mutations_file, chromatin_map_file):
    mutations_file = Path(mutations_file)
    chromatin_map_file = Path(chromatin_map_file)
    # Intersect mutations with chromatin state map
    intersected_file = intersect_mutations_with_map(mutations_file, chromatin_map_file)
    # Count the number of mutations intersecting each region per sample
    intersect_counts_df = count_intersected_file(intersected_file)
    # Count the sizes of each region in the chromatin map
    region_sizes_df = count_region_sizes_in_chrom_map(chromatin_map_file)
    # Normalize the data to mutations per Mb
    normalized_counts_df = normalize_data_to_mutations_per_mb(intersect_counts_df, region_sizes_df)
    # Compare mutation counts between chromatin domains
    domain_comparison_results = compare_domains(normalized_counts_df)
    output_file = intersected_file.with_suffix('.domain_comparison_results.tsv')
    domain_comparison_results.to_csv(output_file, sep='\t')
    # Compare mutation counts between individual chromatin states
    state_comparison_results = compare_all_states(normalized_counts_df)
    output_file = intersected_file.with_suffix('.state_comparison_results.tsv')
    state_comparison_results.to_csv(output_file, sep='\t')
    # Plot the bar graph
    plot_bargraphs(normalized_counts_df)

if __name__ == '__main__':
    mutations_file = 'mutations_file'
    chromatin_map_file = 'chromatin_map_file'
    main(mutations_file, chromatin_map_file)