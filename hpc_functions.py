from cross_correlations import run_cross_corrs
from preprocessing import preprocess_one_session

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--read_path', type=str)
parser.add_argument('--save_folder', type=str)
parser.add_argument('--function', type=str)
parser.add_argument('--pag_i', type=int)

args = parser.parse_args()

if args.function == 'xcorr':
    run_cross_corrs(args.read_path, args.save_folder, False, False, args.pag_i)
elif args.function == 'preprocess':
    preprocess_one_session(args.read_path, args.save_folder, False)