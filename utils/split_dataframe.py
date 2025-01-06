import yaml
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main(): #pylint: disable=too-many-statements
    # read config file
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfgFileNameML.yml', help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    print('Loading analysis configuration: Done!')
    
    fraction = inputCfg["fraction"]
    seed = inputCfg["seed"]
    
    for df_path in inputCfg["dataframes"]:
        df = pd.read_parquet(df_path)
        df_first, df_second = train_test_split(df, test_size=fraction, random_state=seed)
        df_path_name = os.path.splitext(df_path)[0]
        df_first.to_parquet(f"{df_path_name}_frac{fraction}.parquet")
        df_second.to_parquet(f"{df_path_name}_frac{1-fraction}.parquet")

main()