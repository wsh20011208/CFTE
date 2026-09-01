#!/usr/bin/env python3
import argparse
from benchmark_common import add_common_eval_args, evaluate_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FixedCMR with final occlusion-map suppression diagnostics on a gap-specific shared test CSV.')
    add_common_eval_args(parser)
    args = parser.parse_args()
    if args.model_name is None:
        args.model_name = 'fixed_cmr'
    evaluate_csv(args, model_kind='fixed_cmr')
