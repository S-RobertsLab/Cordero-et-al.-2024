from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline, BSpline


def intersect_files(tf_map, mutations, output_name):
    tf_map = Path(tf_map)
    mutations = Path(mutations)
    output_name = mutations.parent / output_name

    # Intersect the files
    with open(output_name, 'w') as output_file:
        subprocess.run(['bedtools', 'intersect', '-wa', '-wb', '-a', tf_map, '-b', mutations], stdout=output_file)

def count_mutations(intersect_file):
    intersect_file = Path(intersect_file)
    counts_file = intersect_file.parent / (intersect_file.stem + '_counts.txt')
    counter = {}

    with open(intersect_file) as f, open(counts_file, 'w') as o:
        for line in f:
            line = line.strip().split('\t')
            # Calculate the midpoint of the interval from the first BED file
            midpoint = int(line[1]) + (int(line[2]) - int(line[1])) / 2

            # Round the midpoint using bankers' rounding
            rounded_midpoint = round(midpoint)

            # Calculate the position of the mutation relative to the midpoint
            relative_position = int(line[8]) - rounded_midpoint

            if line[5] == '-':
                relative_position = -relative_position

            counter[relative_position] = counter.get(relative_position, 0) + 1
        counter = pd.DataFrame.from_dict(counter, orient='index')
        counter.to_csv(o, sep='\t', header=False)
    return counts_file

def plot_mutations(counts_file, show_dots=False):
    plt.style.use('seaborn-whitegrid')  # A clean style for the plot
    plt.rcParams['font.family'] = 'Arial'  # Set a global font family

    # Read in the data from the counts file
    counts_file = Path(counts_file)
    counts = pd.read_csv(counts_file, sep='\t', header=None)
    counts.columns = ['relative_position', 'count']

    # Sort and set index
    counts = counts.sort_values(by='relative_position')

    # Trim the data to only include relative_position from -1000 to 1000
    counts = counts[(counts['relative_position'] >= -1000) & (counts['relative_position'] <= 1000)]

    # Set index after trimming
    counts = counts.set_index('relative_position')
    counts = counts.reindex(np.arange(-1000, 1001), fill_value=0)
    counts = counts.reset_index()
    counts.columns = ['relative_position', 'count']
    counts['relative_position'] = counts['relative_position'].astype(int)

    # Apply the Savitzky-Golay filter
    window_size = 303  # The window size should be an odd number
    poly_order = 7     # The polynomial order is typically a small integer

    # Ensure the 'count' column is in floating-point format for the filter to work correctly
    counts['smoothed_count'] = savgol_filter(counts['count'].astype(np.float64), window_size, poly_order)

    # Now let's create a B-spline representation of the smoothed data
    # First, sort the DataFrame by 'relative_position' to ensure correct spline fitting
    counts_sorted = counts.sort_values('relative_position')

    # Create the spline based on the smoothed data
    spline = make_interp_spline(counts_sorted['relative_position'], counts_sorted['smoothed_count'], k=3)

    # Generate a range of x values for plotting the spline
    x_spline = np.linspace(counts_sorted['relative_position'].min(), counts_sorted['relative_position'].max(), 10000)

    # Compute the spline function on the x values
    y_spline = spline(x_spline)

    # Set up the figure
    plt.figure(figsize=(4, 4))  # Define the size of the figure

    # Optionally add the original data points to the plot
    if show_dots:
        plt.scatter(counts['relative_position'], counts['count'],
                    marker='.', s=10, color='darkgrey', alpha=0.6, label='Raw Data')

    # Plot the spline-interpolated smoothed data
    plt.plot(x_spline, y_spline, color='black', label='Smoothed Curve', linewidth=2)
    plt.ylim([y_spline.min()*0.9, y_spline.max()*1.10])
    # Adjust plot parameters
    plt.xlim(-1000, 1000)
    plt.xlabel('Relative Position (bp)', fontsize=16, fontweight='bold')
    plt.ylabel('Mutation Counts', fontsize=16, fontweight='bold')
    plt.title('Mutations Around TF Binding Midpoint', fontsize=20, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper right', fontsize=12,
                   framealpha=0.5, facecolor='white', edgecolor='black', frameon=True, 
                   borderpad=0.2, labelspacing=0.2, handletextpad=0.2)

    # Tight layout often produces a nicer plot
    plt.tight_layout()

    # Save the plot as a high-resolution PNG or PDF
    # plt.savefig('mutations_plot.png', dpi=300)  # for PNG file
    # plt.savefig('mutations_plot.pdf', format='pdf')  # for PDF file

    plt.show()

def main(tf_map, mutations_bed, output_name):
    output_name = Path(output_name)
    counts_file = Path(output_name.parent / (output_name.stem + '_counts.txt'))
    if not counts_file.exists():
        intersect_files(tf_map, mutations_bed, output_name)
        counts_file = count_mutations(output_name)
        plot_mutations(counts_file, show_dots=True)
    else:
        plot_mutations(counts_file, show_dots=True)

if __name__ == '__main__':
    tf_map = Path('tf_map')
    mutations_bed = Path('mutations')
    output_name = mutations_bed.parent / (mutations_bed.stem + '_filtered_tf.intersect')
    main(tf_map, mutations_bed, output_name)
