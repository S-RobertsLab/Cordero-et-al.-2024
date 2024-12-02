from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ttest_rel
import threading

class Asymmetry():
    def __init__(self, mutation_bed, reference_fasta, transcription_bed, replication_bed, pyrimidine=True, normalization = "mononucleotide"):
        self.mutation_bed = Path(mutation_bed)
        self.reference_fasta = Path(reference_fasta)
        self.transcription_bed = Path(transcription_bed)
        self.replication_bed = Path(replication_bed)
        self.pyrimidine = pyrimidine
        self.normalization = normalization
        self.temp_dir = self.mutation_bed.parent / "temp"
        self.temp_dir.mkdir(exist_ok=True)

    def mutation_combinations(self, mut_type): 
        iupac_trans = {"R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC", "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT", 'A': 'A', 'T': 'T', 'C': 'C', 'G': 'G'}
        from_nuc, to_nuc = mut_type.split('>')
        return [f"{f}>{t}" for f in iupac_trans[from_nuc] for t in iupac_trans[to_nuc] if f != t]

    def context_combinations(self, context):
        iupac_trans = {"R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC", "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGTN", 'A': 'A', 'T': 'T', 'C': 'C', 'G': 'G'}
        return [f"{n1}{n2}{n3}" for n1 in iupac_trans[context[0]] for n2 in iupac_trans[context[1]] for n3 in iupac_trans[context[2]]]

    def reverse_complement(self, seq): 
        complement_table = {"A": "T", "T": "A", "C": "G", "G": "C", "R": "Y", "Y": "R", "S": "S", "W": "W", "K": "M", "M": "K", "B": "V", "D": "H", "H": "D", "V": "B", "N": "N"}
        return "".join(complement_table.get(base, base) for base in seq[::-1])
    
    def reverse_complement_mutation(self, mut_type):
        base1, base2 = mut_type.split('>')
        rev_base1, rev_base2 = self.reverse_complement(base1), self.reverse_complement(base2)
        return f"{rev_base1}>{rev_base2}"
    
    def get_line_count(self, file):
        return int(subprocess.run(f"wc -l {file}", shell=True, check=True, stdout=subprocess.PIPE).stdout.decode().split()[0])

    def bedtools_intersect(self, region_bed_file, mutations_file):
        region_bed_file, mutations_file = Path(region_bed_file), Path(mutations_file)
        intersect_file = self.temp_dir / mutations_file.with_name(f"{mutations_file.stem}_{region_bed_file.stem}.intersect").name
        if intersect_file.exists(): return intersect_file
        subprocess.run(f"bedtools intersect -wa -wb -a {mutations_file} -b {region_bed_file} > {intersect_file}", shell=True, check=True)
        return intersect_file

    def bedtools_get_fasta(self, bed_file, genome_file):
        bed_file, genome_file = Path(bed_file), Path(genome_file)
        fasta_output = bed_file.with_suffix('.fa')
        if fasta_output.exists(): return fasta_output
        with open(fasta_output, 'w') as f: subprocess.run(['bedtools', 'getfasta', '-fi', genome_file, '-bed', bed_file, '-s'], stdout=f)
        return fasta_output

    def adjust_bed_positions(self, bed_file, distance=1):
        bed_file = Path(bed_file); adjusted_bed = self.temp_dir / bed_file.with_name(f"{bed_file.stem}_adjusted.bed").name
        if adjusted_bed.exists(): return adjusted_bed
        with open(bed_file) as f, open(adjusted_bed, 'w') as o:
            o.writelines('\t'.join([line[0], str(int(line[1])-distance), str(int(line[2])+distance)] + line[3:]) + '\n' for line in (l.split('\t') for l in f))
        return adjusted_bed

    def count_contexts(self, bed_file: Path, genome_file):
        trinucleotide_counts = {context: 0 for context in self.context_combinations("NNN")}
        adjusted_bed = self.adjust_bed_positions(bed_file)
        fasta_file = self.bedtools_get_fasta(adjusted_bed, genome_file)
        tri_counts = bed_file.with_suffix('.counts')
        if tri_counts.exists():
            trinucleotide_df = pd.read_csv(tri_counts, index_col=0, header=0, sep='\t')
            return trinucleotide_df
        with open(fasta_file) as f:
            for line in tqdm(f, total=self.get_line_count(fasta_file), unit='lines', desc=f"Counting contexts in {fasta_file.name}"):
                if line.startswith('>'): continue
                line = line.strip().upper()
                for i in range(len(line)-2): trinucleotide_counts[line[i:i+3]] += 1 # count all trinucleotides
        trinucleotide_df = pd.DataFrame.from_dict(trinucleotide_counts, orient='index', columns=['count'])
        trinucleotide_df.to_csv(tri_counts, sep='\t')
        return trinucleotide_df

    def count_intersections(self, mutation_bed_file: Path, region_bed_file: Path, type):
        if type not in ['transcription', 'replication']:
            raise ValueError(f"Type must be either 'transcription' or 'replication'. Got {type}.")
        intersect_file = self.bedtools_intersect(region_bed_file, mutation_bed_file)
        counts = {}
        with open(intersect_file) as f:
            for line in tqdm(f, desc=f"Counting intersections between {mutation_bed_file.name} and {region_bed_file.name}", unit='lines', total=self.get_line_count(intersect_file)):
                line = line.strip().split('\t')
                if type == 'transcription':
                    context, mutation, mut_type, sample, gene_strand = line[6].upper(), line[7], line[8], line[9], line[15]
                    if mut_type != 'SNV': continue
                    if self.pyrimidine:
                        if sample not in counts:
                            counts[sample] = {strand: {mutation: {context: 0 for context in self.context_combinations("NYN")} for mutation in self.mutation_combinations("Y>N")} for strand in ['TS', 'NTS']}
                        # check this logic again, should be consistent
                        if gene_strand == '+':
                            if mutation[0] in 'CT':
                                counts[sample]['NTS'][mutation][context] += 1
                            else:
                                counts[sample]['TS'][self.reverse_complement_mutation(mutation)][self.reverse_complement(context)] += 1
                        else:
                            if mutation[0] in 'CT':
                                counts[sample]['TS'][mutation][context] += 1
                            else:
                                counts[sample]['NTS'][self.reverse_complement_mutation(mutation)][self.reverse_complement(context)] += 1
                    if not self.pyrimidine:
                        if sample not in counts:
                            counts[sample] = {strand: {mutation: {context: 0 for context in self.context_combinations("NRN")} for mutation in self.mutation_combinations("R>N")} for strand in ['TS', 'NTS']}
                        if gene_strand == '+':
                            if mutation[0] in 'AG':
                                counts[sample]['NTS'][mutation][context] += 1
                            else:
                                counts[sample]['TS'][self.reverse_complement_mutation(mutation)][self.reverse_complement(context)] += 1
                        else:
                            if mutation[0] in 'AG':
                                counts[sample]['TS'][mutation][context] += 1
                            else:
                                counts[sample]['NTS'][self.reverse_complement_mutation(mutation)][self.reverse_complement(context)] += 1
                # fill this in later
                elif type == 'replication':
                    context, mutation, mut_type, sample, replication_direction = line[6].upper(), line[7], line[8], line[9], line[13]
                    if mut_type != 'SNV': continue
                    if self.pyrimidine:
                        if sample not in counts:
                            counts[sample] = {rep_strand: {mutation: {context: 0 for context in self.context_combinations("NYN")} for mutation in self.mutation_combinations("Y>N")} for rep_strand in ['Leading', 'Lagging']}
                        if replication_direction == 'right':
                            if mutation[0] in 'CT':
                                counts[sample]['Lagging'][mutation][context] += 1
                            else:
                                counts[sample]['Leading'][self.reverse_complement_mutation(mutation)][self.reverse_complement(context)] += 1
                        else:
                            if mutation[0] in 'CT':
                                counts[sample]['Leading'][mutation][context] += 1
                            else:
                                counts[sample]['Lagging'][self.reverse_complement_mutation(mutation)][self.reverse_complement(context)] += 1
                    if not self.pyrimidine:
                        if sample not in counts:
                            counts[sample] = {rep_strand: {mutation: {context: 0 for context in self.context_combinations("NRN")} for mutation in self.mutation_combinations("R>N")} for rep_strand in ['Leading', 'Lagging']}
                        if replication_direction == 'right':
                            if mutation[0] in 'AG':
                                counts[sample]['Lagging'][mutation][context] += 1
                            else:
                                counts[sample]['Leading'][self.reverse_complement_mutation(mutation)][self.reverse_complement(context)] += 1
                        else:
                            if mutation[0] in 'AG':
                                counts[sample]['Leading'][mutation][context] += 1
                            else:
                                counts[sample]['Lagging'][self.reverse_complement_mutation(mutation)][self.reverse_complement(context)] += 1
        return counts
    
    def normalize_counts(self, intersection_counts, context_counts, normalization):
        if normalization not in ['mononucleotide', 'trinucleotide']:
            raise ValueError(f"Normalization must be either 'mononucleotide' or 'trinucleotide'. Got {normalization}.")
        
        normalized_data = {}

        # Mononucleotide normalization: sum by middle nucleotide (e.g., C for C>A)
        if normalization == 'mononucleotide':
            # Sum context counts by the middle nucleotide
            mono_context_counts = context_counts.groupby(context_counts.index.str[1]).sum() 

            # Iterate over each sample
            for sample, data in intersection_counts.items():
                for strand, mutations in data.items():
                    for mutation, contexts in mutations.items():
                        for context, count in contexts.items():
                            middle_base = context[1] if strand in ['NTS', 'Leading'] else self.reverse_complement(context[1])
                            if middle_base == 'N':  # Skip ambiguous contexts
                                continue
                            # Normalize by the total count of this middle base
                            total_middle_base_count = mono_context_counts.loc[middle_base, 'count']
                            normalized_value = count / total_middle_base_count
                            normalized_data.setdefault(sample, {}).setdefault(strand, {}).setdefault(mutation, {})[context] = normalized_value

        # Trinucleotide normalization: divide by each specific trinucleotide context count
        elif normalization == 'trinucleotide':
            for sample, data in intersection_counts.items():
                for strand, mutations in data.items():
                    for mutation, contexts in mutations.items():
                        for context, count in contexts.items():
                            # Normalize by the specific trinucleotide count
                            tri_context_count = context_counts.loc[context, 'count'] if strand in ['NTS', 'Leading'] else context_counts.loc[self.reverse_complement(context), 'count']
                            normalized_value = count / tri_context_count
                            normalized_data.setdefault(sample, {}).setdefault(strand, {}).setdefault(mutation, {})[context] = normalized_value

        # Sum normalized values for each mutation type across all contexts
        summed_normalized_data = {}

        for sample, strands in normalized_data.items():
            for strand, mutations in strands.items():
                for mutation, contexts in mutations.items():
                    # Sum all normalized values for this mutation type
                    total_mutation_sum = sum(contexts.values())
                    # Store the summed value in the new dictionary
                    summed_normalized_data.setdefault(sample, {}).setdefault(strand, {})[mutation] = total_mutation_sum            

        return summed_normalized_data

    def plot_transcription_data(self, normalized_counts, normalization):
        """
        Plots the transcription data with error bars, replicate scatter points, and performs paired t-tests.
        
        Parameters:
        normalized_counts (dict): Normalized mutation counts for samples.
        normalization (str): The normalization method used ('mononucleotide' or 'trinucleotide').
        """

        # Set Seaborn whitegrid style but limit gridlines to the y-axis (horizontal gridlines only)
        sns.set(style="whitegrid", rc={'axes.grid.axis': 'y', 'axes.grid': True})

        # Override edgecolor for axes lines to ensure they are black
        plt.rcParams['axes.edgecolor'] = 'black'

        # Organize data
        mutation_types = normalized_counts[list(normalized_counts.keys())[0]]['TS'].keys()
        if not self.pyrimidine:
            # Reverse the order of the mutation types
            mutation_types = list(mutation_types)[::-1]
        samples = list(normalized_counts.keys())

        # Data storage for plotting
        transcribed_values = []
        non_transcribed_values = []
        p_values = []

        # Organize the data and normalize individual values
        for mutation in mutation_types:
            ts_vals = []
            nts_vals = []
            for sample in samples:
                ts_vals_sample = normalized_counts[sample]['TS'].get(mutation, 0)
                nts_vals_sample = normalized_counts[sample]['NTS'].get(mutation, 0)

                # Apply the selected normalization to individual values
                if normalization == 'mononucleotide':
                    ts_vals_sample *= 1e6
                    nts_vals_sample *= 1e6
                ts_vals.append(ts_vals_sample)
                nts_vals.append(nts_vals_sample)

            transcribed_values.append(ts_vals)
            non_transcribed_values.append(nts_vals)

            # Perform paired t-test comparing TS vs NTS for each mutation type
            t_stat, p_value = ttest_rel(ts_vals, nts_vals)  # Paired t-test
            p_values.append(p_value)  # Store the p-value

        # Convert the values to arrays for easier manipulation
        transcribed_means = np.mean(transcribed_values, axis=1)
        transcribed_stds = np.std(transcribed_values, axis=1)
        non_transcribed_means = np.mean(non_transcribed_values, axis=1)
        non_transcribed_stds = np.std(non_transcribed_values, axis=1)

        # For trinucleotide normalization, divide by the median
        if normalization == 'trinucleotide':
            median_ts = np.median([val for sublist in transcribed_values for val in sublist])
            median_nts = np.median([val for sublist in non_transcribed_values for val in sublist])

            transcribed_values = np.array(transcribed_values) / median_ts
            non_transcribed_values = np.array(non_transcribed_values) / median_nts
            transcribed_means = transcribed_means / median_ts
            non_transcribed_means = non_transcribed_means / median_nts
            y_label = 'Normalized Values'
        else:
            y_label = 'Mutations per Mb'

        # Create a figure and axes with specific layout settings
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})

        # Define the width of the bars and the positions
        width = 0.35  # the width of the bars
        ind = np.arange(len(mutation_types))  # the x locations for the groups

        # Plotting the bars (transcribed and non_transcribed)
        p1 = ax.bar(ind - width/2, non_transcribed_means, width, yerr=non_transcribed_stds, color='IndianRed', 
                    label='Non-Transcribed', edgecolor='black', linewidth=1.5, capsize=5)
        p2 = ax.bar(ind + width/2, transcribed_means, width, yerr=transcribed_stds, color='SkyBlue', 
                    label='Transcribed', edgecolor='black', linewidth=1.5, capsize=5)

        # Plot individual normalized values (scatter dots over bars) only if neither value is zero
        for i, mutation in enumerate(mutation_types):
            ts_vals = transcribed_values[i]
            nts_vals = non_transcribed_values[i]

            if np.sum(ts_vals) > 0 and np.sum(nts_vals) > 0:  # Only plot dots if both strands have values
                jitter_ts = np.linspace(-width/8, width/8, len(ts_vals))
                jitter_nts = np.linspace(-width/8, width/8, len(nts_vals))

                ax.scatter(ind[i] + jitter_ts + width/2, ts_vals, color='SkyBlue', alpha=0.7, edgecolor='black', linewidths=2)
                ax.scatter(ind[i] + jitter_nts - width/2, nts_vals, color='IndianRed', alpha=0.7, edgecolor='black', linewidths=2)

        # Customizing the ax (main bar plot)
        ax.set_xticks(ind)
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xticklabels(mutation_types, fontsize=12)  # Rotation for better readability
        ax.set_title('Transcriptional Strand Asymmetry', fontsize=20, weight='bold')
        ax.set_ylabel(y_label, fontsize=16, weight='bold')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.5, 
                facecolor='white', edgecolor='black', frameon=True, 
                borderpad=0.2, labelspacing=0.2, handletextpad=0.2)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Remove the top and right spines for the top plot only (ax)
        sns.despine(ax=ax, top=True, right=True)

        # Customizing the ax2 (log2 ratios)
        log2_ratios = []
        for i in range(len(transcribed_means)):
            if transcribed_means[i] > 0 and non_transcribed_means[i] > 0:
                log2_ratio = np.log2(non_transcribed_means[i] / transcribed_means[i])
            else:
                log2_ratio = 0  # Set log2 ratio to 0 if either value is 0
            log2_ratios.append(log2_ratio)

        log2_ratios = np.array(log2_ratios)
        colors = ['SkyBlue' if ratio < 0 else 'IndianRed' for ratio in log2_ratios]  # Flipped colors based on the ratio
        ax2.bar(ind, log2_ratios, color=colors, edgecolor='black', linewidth=1)
        ax2.axhline(y=0, color='black', linewidth=1.5)  # Adding horizontal line at y=0
        ax2.set_xticks(ind)
        ax2.set_xticklabels(mutation_types, fontsize=12)  # Consistency in labels with the main plot
        ax2.set_ylabel('Log2 Ratio', fontsize=16, weight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.set_ylim([-1, 1])

        # Output p-values to the console
        print("\nT-Test P-Values for Transcriptional Strand Asymmetry (Paired):")
        for mutation, p_value in zip(mutation_types, p_values):
            print(f"P-value for {mutation} (NTS vs TS): {p_value:.10f}")

        # Show the plot
        plt.show()

    def plot_replication_data(self, normalized_counts, normalization):
        """
        Plots the strand data (leading vs lagging) with error bars, replicate scatter points, and performs paired t-tests.
        
        Parameters:
        normalized_counts (dict): Normalized mutation counts for samples.
        normalization (str): The normalization method used ('mononucleotide' or 'trinucleotide').
        """

        # Set Seaborn whitegrid style but limit gridlines to the y-axis (horizontal gridlines only)
        sns.set(style="whitegrid", rc={'axes.grid.axis': 'y', 'axes.grid': True})

        # Override edgecolor for axes lines to ensure they are black
        plt.rcParams['axes.edgecolor'] = 'black'

        # Organize data
        mutation_types = normalized_counts[list(normalized_counts.keys())[0]]['Leading'].keys()
        if not self.pyrimidine:
            # Reverse the order of the mutation types
            mutation_types = list(mutation_types)[::-1]
        samples = list(normalized_counts.keys())

        # Data storage for plotting
        leading_values = []
        lagging_values = []
        p_values = []

        # Organize the data and normalize individual values
        for mutation in mutation_types:
            leading_vals = []
            lagging_vals = []
            for sample in samples:
                leading_vals_sample = normalized_counts[sample]['Leading'].get(mutation, 0)
                lagging_vals_sample = normalized_counts[sample]['Lagging'].get(mutation, 0)

                # Apply the selected normalization to individual values
                if normalization == 'mononucleotide':
                    leading_vals_sample *= 1e6
                    lagging_vals_sample *= 1e6
                leading_vals.append(leading_vals_sample)
                lagging_vals.append(lagging_vals_sample)

            leading_values.append(leading_vals)
            lagging_values.append(lagging_vals)

            # Perform paired t-test comparing leading vs lagging for each mutation type
            t_stat, p_value = ttest_rel(leading_vals, lagging_vals)
            p_values.append(p_value)  # Store the p-value

        # Convert the values to arrays for easier manipulation
        leading_means = np.mean(leading_values, axis=1)
        leading_stds = np.std(leading_values, axis=1)
        lagging_means = np.mean(lagging_values, axis=1)
        lagging_stds = np.std(lagging_values, axis=1)

        # For trinucleotide normalization, divide by the median
        if normalization == 'trinucleotide':
            median_leading = np.median([val for sublist in leading_values for val in sublist])
            median_lagging = np.median([val for sublist in lagging_values for val in sublist])

            leading_values = np.array(leading_values) / median_leading
            lagging_values = np.array(lagging_values) / median_lagging
            leading_means = leading_means / median_leading
            lagging_means = lagging_means / median_lagging
            y_label = 'Normalized Values'
        else:
            y_label = 'Mutations per Mb'

        # Create a figure and axes with specific layout settings
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})

        # Define the width of the bars and the positions
        width = 0.35  # the width of the bars
        ind = np.arange(len(mutation_types))  # the x locations for the groups

        # Plotting the bars (leading and lagging), leading first
        p1 = ax.bar(ind - width/2, leading_means, width, yerr=leading_stds, color='IndianRed', 
                    label='Leading Strand', edgecolor='black', linewidth=1.5, capsize=5)
        p2 = ax.bar(ind + width/2, lagging_means, width, yerr=lagging_stds, color='SkyBlue', 
                    label='Lagging Strand', edgecolor='black', linewidth=1.5, capsize=5)

        # Plot individual normalized values (scatter dots over bars) only if neither value is zero
        for i, mutation in enumerate(mutation_types):
            leading_vals = leading_values[i]
            lagging_vals = lagging_values[i]

            if np.sum(leading_vals) > 0 and np.sum(lagging_vals) > 0:  # Only plot dots if both strands have values
                jitter_leading = np.linspace(-width/8, width/8, len(leading_vals))
                jitter_lagging = np.linspace(-width/8, width/8, len(lagging_vals))

                ax.scatter(ind[i] + jitter_leading - width/2, leading_vals, color='IndianRed', alpha=0.7, edgecolor='black', linewidths=2)
                ax.scatter(ind[i] + jitter_lagging + width/2, lagging_vals, color='SkyBlue', alpha=0.7, edgecolor='black', linewidths=2)

        # Customizing the ax (main bar plot)
        ax.set_xticks(ind)
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xticklabels(mutation_types, fontsize=12)  # Rotation for better readability
        ax.set_title('Replication Strand Asymmetry', fontsize=20, weight='bold')
        ax.set_ylabel(y_label, fontsize=16, weight='bold')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.5, 
                facecolor='white', edgecolor='black', frameon=True, 
                borderpad=0.2, labelspacing=0.2, handletextpad=0.2)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Remove the top and right spines for the top plot only (ax)
        sns.despine(ax=ax, top=True, right=True)

        # Customizing the ax2 (log2 ratios)
        log2_ratios = []
        for i in range(len(leading_means)):
            if leading_means[i] > 0 and lagging_means[i] > 0:
                log2_ratio = np.log2(leading_means[i] / lagging_means[i])
            else:
                log2_ratio = 0  # Set log2 ratio to 0 if either value is 0
            log2_ratios.append(log2_ratio)

        log2_ratios = np.array(log2_ratios)
        colors = ['SkyBlue' if ratio < 0 else 'IndianRed' for ratio in log2_ratios]  # Flipped colors based on the ratio
        ax2.bar(ind, log2_ratios, color=colors, edgecolor='black', linewidth=1)
        ax2.axhline(y=0, color='black', linewidth=1.5)  # Adding horizontal line at y=0
        ax2.set_xticks(ind)
        ax2.set_xticklabels(mutation_types, fontsize=12)  # Consistency in labels with the main plot
        ax2.set_ylabel('Log2 Ratio', fontsize=16, weight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.set_ylim([-1, 1])

        # Output p-values to the console
        print("\nT-Test P-Values for Leading vs Lagging Strand Asymmetry (Paired):")
        for mutation, p_value in zip(mutation_types, p_values):
            print(f"P-value for {mutation} (Leading vs Lagging): {p_value:.10f}")

        # Show the plot
        plt.show()

    def main(self):
        results = {}
        lock = threading.Lock()  # To manage access to the shared 'results' dict

        # Function to get context counts and store results in a single dataframe
        def get_contexts(bed, key_prefix):
            try:
                context_data = self.count_contexts(bed, self.reference_fasta)
                with lock:  # Ensure thread-safe access to 'results'
                    results[f'{key_prefix}_contexts'] = context_data
            except Exception as e:
                print(f"Error in {key_prefix} thread: {e}")
                return None

        # Function to perform the intersection and store counts
        def intersect_contexts(mutation_file, region_file, key_prefix):
            try:
                intersection_data = self.count_intersections(mutation_file, region_file, key_prefix)
                with lock:  # Ensure thread-safe access to 'results'
                    results[f"{key_prefix}_intersection"] = intersection_data
            except Exception as e:
                print(f"Error in intersection thread ({key_prefix}): {e}")
                return None

        # Create threads for transcription and replication contexts
        threads = [
            threading.Thread(target=get_contexts, args=(self.transcription_bed, 'transcription')),
            threading.Thread(target=get_contexts, args=(self.replication_bed, 'replication'))
        ]

        # Add threads for intersection
        threads += [
            threading.Thread(target=intersect_contexts, args=(self.mutation_bed, self.transcription_bed, 'transcription')),
            threading.Thread(target=intersect_contexts, args=(self.mutation_bed, self.replication_bed, 'replication'))
        ]

        # Start and join all threads
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Access the context results
        transcription_context_counts = results.get('transcription_contexts')
        replication_context_counts = results.get('replication_contexts')

        # Check intersection results
        transcription_intersection_counts = results.get('transcription_intersection')
        replication_intersection_counts = results.get('replication_intersection')

        if transcription_context_counts is None:
            print("Transcription context data not available.")
        if replication_context_counts is None:
            print("Replication context data not available.")
        if transcription_intersection_counts is None:
            print("Transcription intersection data not available.")
        if replication_intersection_counts is None:
            print("Replication intersection data not available.")

        normalized_transcription_data = self.normalize_counts(transcription_intersection_counts, transcription_context_counts, self.normalization)
        normalized_replication_data = self.normalize_counts(replication_intersection_counts, replication_context_counts, self.normalization)
        ###########################################################
        # write the data to a tsv file
        # here is the data
        import csv

        # Function to reshape and write the data
        def write_data_reshaped(data, data_type):
            # Determine all unique mutation types across the dataset
            mutation_types = set()
            for strands in data.values():
                for mutations in strands.values():
                    mutation_types.update(mutations.keys())
            mutation_types = sorted(mutation_types)  # Sort for consistent column order

            # File to write the data
            output_file = f"/media/cam/Working/Data_Table/{data_type}_reshaped_data.tsv"

            # Write reshaped data to the file
            with open(output_file, "w", newline="") as tsv_file:
                writer = csv.writer(tsv_file, delimiter="\t")
                # Write header
                header = ["Sample", "Strand"] + mutation_types
                writer.writerow(header)

                # Write rows
                for sample, strands in data.items():
                    for strand, mutations in strands.items():
                        row = [sample, strand] + [mutations.get(mutation, 0) * 1e6 for mutation in mutation_types]
                        writer.writerow(row)

            print(f"Data written to {output_file}")


        # Write both transcription and replication data
        for data_type, data in zip(
            ["transcription", "replication"],
            [normalized_transcription_data, normalized_replication_data],
        ):
            write_data_reshaped(data, data_type)


        ###########################################################
        self.plot_transcription_data(normalized_transcription_data, self.normalization)
        self.plot_replication_data(normalized_replication_data, self.normalization)

        # Now return or process the results as needed
        return results


Asymmetry(  'mutations.bed',
            'genome.fa',
            'ensembl_gene_list',
            'replication_map',
            pyrimidine=False, 
            normalization='mononucleotide'
        ).main()