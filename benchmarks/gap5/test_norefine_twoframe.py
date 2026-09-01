#!/usr/bin/env python3
import argparse
from benchmark_common import add_common_eval_args, evaluate_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate NoRefine with final occlusion-map suppression diagnostics on a gap-specific shared test CSV.')
    add_common_eval_args(parser)
    args = parser.parse_args()
    if args.model_name is None:
        args.model_name = 'norefine_twoframe'
    evaluate_csv(args, model_kind='norefine_twoframe')
