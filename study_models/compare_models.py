import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
import ast

parser = argparse.ArgumentParser(description="Process ML analysis with pt bins and data files.")
parser.add_argument("firstdf", type=str, help="Path to the first data file.")
parser.add_argument("firstdf_name", type=str, help="Name/label for the first data file.")
parser.add_argument("secdf", type=str, help="Path to the second data file.")
parser.add_argument("secdf_name", type=str, help="Name/label for the second data file.")
parser.add_argument("-o", "--out_file", type=str, required=True, help="Path to the output file.")
parser.add_argument("-pt", "--pt_bins", type=str, required=True, help="List of pt bins in the format: '[1,3,5,8,12,24]'.")

args = parser.parse_args()
dfs_folders = [args.firstdf, args.secdf]  
dfs_labels = [args.firstdf_name, args.secdf_name]
out_folder = args.out_file
pt_bins = ast.literal_eval(args.pt_bins)
pt_bins = list(zip(pt_bins[:-1], pt_bins[1:]))

classes = ["Bkg", "Prompt", "NonPrompt"]
label_classes = ["Bkg", "Prompt", "FD"]
label_colors = ['#1f77b4', '#ff7f0e', '#9467bd']

nbins = 100
bins = np.linspace(0,1,101)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_widths = bins[1:] - bins[:-1]

nbins = 100
bins = np.linspace(0, 1, nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_widths = bins[1:] - bins[:-1]

for ipt in pt_bins:
    print(f'ipt: {ipt}')
    for iclass in classes:
        print(f'iclass: {iclass}')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        ratio_maxs = []
        ratio_mins = []
        for label_class, label_color in zip(label_classes, label_colors):
            histos_counts = []
            for ifolder, (ifolder_name, idf_label) in enumerate(zip(dfs_folders, dfs_labels)):
                df = pd.read_parquet(
                    f"{ifolder_name}/pt{ipt[0]}_{ipt[1]}/{iclass}_pT_{ipt[0]}_{ipt[1]}_ModelApplied.parquet.gzip"
                )

                # Calculate histogram counts and append
                counts, edges = np.histogram(df[f"ML_output_{label_class}"], bins=bins)
                histos_counts.append(counts)
                integral = np.sum(counts)

                # Plot the markers on top of the histogram
                if ifolder == 0:
                    ax1.plot(bin_centers, counts, 'o', color=label_color, label=f"{idf_label}_{label_class}")
                else:
                    # Fill the area under the histogram
                    ax1.bar(bin_centers, counts, width=bin_widths, color=label_color, alpha=0.5,
                            label=f"{idf_label}_{label_class}", log=True)

            ratios = np.array(histos_counts[1]) / np.array(histos_counts[0])
            ratios_unc = np.sqrt( (np.array(np.sqrt(histos_counts[1])) / np.array(histos_counts[0]))**2 + 
                                 ( (np.array(histos_counts[1])*np.array(np.sqrt(histos_counts[0]))) / np.array(histos_counts[0])**2)**2 )
            ax2.errorbar(bin_centers, ratios, yerr=ratios_unc, xerr=bin_widths, label=f'{label_class}', fmt='o', color=label_color)
            
            max_ratio_idx = np.argmax([ratio + ratio_unc for ratio, ratio_unc in zip(ratios, ratios_unc)])
            ratio_maxs.append(ratios[max_ratio_idx] + ratios_unc[max_ratio_idx])
            min_ratio_idx = np.argmin([ratio - ratio_unc for ratio, ratio_unc in zip(ratios, ratios_unc)])
            ratio_mins.append(ratios[min_ratio_idx] - ratios_unc[max_ratio_idx])

        ax1.set_title(f"Distributions of ML Outputs for {iclass}, pt_{ipt[0]}_{ipt[1]}", fontsize=18)
        ax1.set_xlabel("Score")
        ax1.set_ylabel("Frequency (log scale)")
        ax1.set_yscale('log')
        ax1.legend()

        ax2.set_xlabel("Score")
        ax2.set_ylabel(f"{dfs_labels[1]}/{dfs_labels[0]}")
        # ax2.set_ylim([min(ratio_mins)-min(ratio_mins)*0.001,max(ratio_maxs)+max(ratio_maxs)*0.1])
        ax2.grid()
        # ax2.set_yscale('log')
        ax2.legend()

        if not os.path.isdir(os.path.expanduser(f"{out_folder}/compare/pt_{ipt[0]}_{ipt[1]}/")):
            os.makedirs(os.path.expanduser(f"{out_folder}/compare/pt_{ipt[0]}_{ipt[1]}/"))
        plt.savefig(
            f"{out_folder}/compare/pt_{ipt[0]}_{ipt[1]}/{iclass}Distrs_3050_50100.pdf",
            format="pdf",
            bbox_inches="tight",
        )
