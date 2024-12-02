from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def reverse_complement(seq: str):
    '''
    Takes a nucleotide string in IUPAC and regular format and returns the reverse complement
    '''
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A',
                        "R":"Y", "Y":"R", "S":"S", "W":"W", "K":"M",
                        "M":"K","B":"V", "D":"H", "H":"D", "V":"B",
                        "N":"N"}
    return "".join(complement.get(base, base) for base in reversed(seq))

def count_flanking_bases(input_filename):
    input_path = Path(input_filename)
    output_path = input_path.with_stem(input_path.stem + '_flanking_base_counts').with_suffix('.tsv')
    with open(input_path, 'r') as f:
        flank_counts = {i: {} for i in range(-5, 6)}
        for line in f:
            line = line.strip().split('\t')
            context = line[6].upper()
            if context[7:13] == 'TTTTTT':
                continue
            if context[7] == 'T' and len(context) == 14:
                for i in range(-5, 6):
                    base = context[i+7]
                    flank_counts[i][base] = flank_counts[i].get(base, 0) + 1

    t_data = pd.DataFrame(flank_counts).fillna(0).astype(int)
    t_data.to_csv(output_path, sep='\t')
    return t_data

def graph_flanking_bases(input_filename):
    # Set the style of the plot
    sns.set(style="whitegrid")

    # Count the flanking bases
    t_data = count_flanking_bases(input_filename)

    # Transpose the DataFrame and reorder the columns
    t_data = t_data.T[['T', 'A', 'C', 'G']]

    # Create the bar plot
    ax = t_data.plot(kind='bar', stacked=True, colormap='viridis', edgecolor='black')

    # Set the figure size
    ax.figure.set_size_inches(4, 4)

    # Add a title and labels
    ax.set_title('Distribution of Flanking Bases', fontsize=16, fontweight='bold')
    ax.set_xlabel('Position', fontsize=16, weight='bold')
    ax.set_ylabel('Count', fontsize=16, weight='bold')

    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Bases', title_fontsize='16', fontsize='16')

    # Remove the vertical gridlines
    ax.xaxis.grid(False)

    # Make the x-axis labels horizontal and bold
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=16, weight='bold')

    # Remove the top and right spines
    sns.despine()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    input_filename = 'filename'
    graph_flanking_bases(input_filename)