"""
Simple example algorithm to use in lieu of the DL model for evaluation

Assume the project directory hierarchy:

prj-dir
|--> ####
  |--> ctp_auxiliary_slice_#.npz

Where each file ctp_auxiliary_slice_#.npz contains an object with the key 
'tmax' that evaluates to a numpy array of size (416, 416).
"""

import argparse
import os
import re
import numpy as np

def _get_parser():
    """Get the parser to parse the command line arguments.

    Returns
    -------
    argparse.ArgumentParser
        The configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="Use the preprocessed Tmax maps for lesion segmentation")
    parser.add_argument(
        "-p",
        "--prj_dir",
        type=str,
        required=True,
        help="Path to the project directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=False,
        help="Path to the output directory. Defaults to the value of -p",
    )
    parser.add_argument(
        "-s",
        "--sbj_id",
        type=str,
        nargs='+',
        required=False,
        help="Four-digit numerical subject IDs to process, delimited by spaces (e.g. 0003 0004)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        action='store_true',
        help="Process all subjects in the project directory. Use in lieu of -s",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true'
    )
    return parser

def tmax_threshold(in_dir, out_dir, verbose):
    # Load the Tmax values from each slice into a 3D (ZXY) volume. The atypical
    # axis ordering emulates the dimensions of a batch of slices in a DL model
    in_files = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]
    tmax_files = []
    for f in in_files:
        match = re.match(r'^.+ctp_auxiliary_slice_(\d+).npz$', f)
        if match is not None:
            tmax_files.append(match)
    tmax_files = [m[0] for m in sorted(tmax_files, key=lambda m: int(m[1]))]
    if verbose:
        print(f'Reading {len(tmax_files)} slice files')
    tmax_map = np.stack([np.load(f)['tmax'] for f in tmax_files], axis=0)

    # Apply the thresholding operation
    tmax_map = tmax_map>6

    # Save the result as if it had come from a DL model
    predictions = np.stack((~tmax_map, tmax_map), axis=-1)
    out_path = os.path.join(out_dir, 'model_prediction.npz')
    if verbose:
        print(f'Writing lesion prediction to  {out_path}')
    np.savez_compressed(out_path, pred=predictions)


def main(args=None):
    """entry point."""

    options = _get_parser().parse_args(args)
    # Parse options to determine which subject IDs to process
    if options.batch:
        # Discover the subject data, unless subject IDs were also provided, which would be invalid.
        if options.sbj_id:
            print('ERROR: Only use one of -s and -b')
            exit()
        # For interactive explanations of the regex patterns, see https://regex101.com/
        sbj_dir_pattern = r"^\d{4}$"
        sbj_dirs = os.listdir(options.prj_dir)
        options.sbj_id = []
        for sbj_dir in sbj_dirs:
            match = re.match(sbj_dir_pattern, sbj_dir)
            if match:
                options.sbj_id.append(match[0])
        if options.verbose:
            print(f"Discovered the following subject directories in {options.prj_dir}:")
            for s in options.sbj_id:
                print(s)
    # Require one of -s or -b. Otherwise, there are no sbj_id to process
    elif not options.sbj_id:
        print('ERROR: Use one of -s or -b')
        exit()

    # Process each subject ID:
    for s in options.sbj_id:
        in_dir = os.path.join(options.prj_dir,
                              s)

        if options.out_dir:
            out_dir = os.path.join(options.out_dir, s)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
        else:
            out_dir = os.path.join(options.prj_dir,
                                   s)
        if options.verbose:
            print(f"Now processing {s}")

        tmax_threshold(in_dir, 
                       out_dir,
                       options.verbose)

        if options.verbose:
            print(f"End processing {s}\n")

if __name__ == "__main__":
    main()
