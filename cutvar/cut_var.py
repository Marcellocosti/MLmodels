import ROOT
from ROOT import TFile
import multiprocessing as mp
import itertools
import sys
import yaml
import argparse

sys.path.append('/home/mdicosta/alice/DmesonAnalysis/run3/flow')
from flow_analysis_utils import get_vn_versus_mass

def process_sel(sel, var_ax_numbers, var_names, sparse, proj_vn=False, inv_mass_bins=[]):
    thn_sparse = sparse.Clone()
    directory_path = ''
    for i, var in enumerate(sel):
        if var[1] > var[0]:
            directory_path += f'{var_names[i]}_{var[0]}_{var[1]}/'
            thn_sparse.GetAxis(var_ax_numbers[i]).SetRangeUser(var[0], var[1])
    
    print(directory_path)
    proj_histos = []
    for axis in var_ax_numbers:
        histo = thn_sparse.Projection(axis)
        histo.SetDirectory(0)
        proj_histos.append(histo)
    
    if proj_vn:
        mass_axis = next((i for i, str in enumerate(var_names) if 'mass' in str), -1)
        hist_vn = get_vn_versus_mass(thn_sparse, inv_mass_bins, mass_axis, var_ax_numbers[-1], False)
        hist_vn.SetDirectory(0)
        proj_histos.append(hist_vn)

    return directory_path, proj_histos

# Parallelize over pt_bins
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process selection variables from YAML configuration.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration from the specified YAML file
    with open(args.yaml_path, "r") as file:
        config = yaml.safe_load(file)

    sparse_file = TFile.Open(config['input_file'])
    sparse = sparse_file.Get(config['sparse_path'])

    var_names, var_ax_numbers, var_ranges = [], [], []
    for var in config['vars']:
        var_names.append(var['name'])
        var_ax_numbers.append(var['axis'])
        if var.get('bins'):
            var_ranges.append(list(zip(var['bins'][:-1], var['bins'][1:])))
        elif var.get('upp_vals'):
            first_bin = sparse.GetAxis(var['axis']).GetBinLowEdge(1)
            var_ranges.append([[first_bin, upper_val] for upper_val in var['upp_vals']])
        elif var.get('low_vals'):
            last_bin = sparse.GetAxis(var['axis']).GetBinLowEdge(sparse.GetAxis(var['axis']).GetNbins()) + sparse.GetAxis(var['axis']).GetBinWidth(1)
            var_ranges.append([[low_val, last_bin] for low_val in var['low_vals']])
        else:
            var_ranges.append([[0., -1]])
    
    if config.get('proj_vn'):
        var_names.append('vn')
        var_ax_numbers.append('vn_axis')
        
    inv_mass_bins = config.get('inv_mass_bins', [])

    sels = list(itertools.product(*var_ranges))
    outFile = TFile(config['output_file'], 'recreate')  # Use 'update' mode to prevent overwriting
    args = [(sel, var_ax_numbers, var_names, sparse, config.get('proj_vn'), inv_mass_bins) for sel in sels]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        var_histos = pool.starmap(process_sel, args)
        
    print('Saving histograms to output file...')
    for directory_path, histograms in var_histos:
        outFile.mkdir(directory_path)
        outFile.cd(directory_path)
        for histo, var_name in zip(histograms, var_names):
            histo.Write(f"h{var_name}")

    outFile.Close()
