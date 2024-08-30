"""
Evaluation script for the ISLES24 challenge lesion segmentations
Assume the project directory hierarchy:

prj-dir
|--> ####
  |--> lesion_segmentation.nii.gz

Where the file model_prediction.npz contains an object with the key 'pred'
that evaluates to a numpy array of size (slices, 416, 416, 2). The last
element of the inner dimension should be the voxelwise infarct probabilities
which fall in the range [0, 1]
"""

import argparse
import os
import re
import csv
import numpy as np
import SimpleITK as sitk
# Local imports:
import eval_utils
import helpers

def _get_parser():
    """Get the parser to parse the command line arguments.

    Returns
    -------
    argparse.ArgumentParser
        The configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="Preprocess the ISLES data")
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

# Define true Dice
def dsc(a, b):
    _a = np.array(a).astype(bool)
    _b = np.array(b).astype(bool)
    return 2*np.sum(_a*_b)/(np.sum(_a)+np.sum(_b))

def evaluate(in_dir, verbose):
    # Get a 3D numpy array of image scalars somehow
    prediction_path = os.path.join(in_dir, 'lesion_segmentation.nii.gz')
    ground_truth_path = os.path.join(in_dir, 'ctp_seg_original.nii.gz')

    if verbose:
        print(f'Reading model prediction from {prediction_path}')
    prediction = sitk.ReadImage(prediction_path)
    prediction = helpers.image_to_ndarr(prediction)

    if verbose:
        print(f'Reading ground_truth from {ground_truth_path}')
    ground_truth = sitk.ReadImage(ground_truth_path)
    voxel_volume = np.prod(ground_truth.GetSpacing())
    ground_truth = helpers.image_to_ndarr(ground_truth)

    # IOU metrics are computed with an intersetction-over-union threshold of
    # 0.5. Below this threshold, ground-truth connected components are scored
    # as having no true-positives. F1 and Dice are computed here, as well as
    # the abs. value of the difference in the number of connected components
    iou_f1, abs_lesion_ct_diff, iou_dice = eval_utils.compute_dice_f1_instance_difference(ground_truth, prediction)
    # The challenge is meant to use the global dice score instead of IOU, so
    # we compute it with our own function. 
    dice = dsc(ground_truth, prediction)
    # Absolute volume difference depends on the voxel size as computed above
    abs_vol_diff = eval_utils.compute_absolute_volume_difference(ground_truth, prediction, voxel_volume)

    return iou_f1, abs_lesion_ct_diff, iou_dice, dice, abs_vol_diff


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

    if not os.path.isdir(options.out_dir):
        os.makedirs(options.out_dir)
    
    out_file = os.path.join(options.out_dir, 'evaluation_metrics_casewise.csv')
    with open(out_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        header = ["subj_id", "iou_f1", "abs_lesion_ct_diff", "iou_dice", "dice", "abs_vol_diff"]
        writer.writerow(header)

        # Process each subject ID:
        metrics_agg = []
        for s in options.sbj_id:
            in_dir = os.path.join(options.prj_dir, s)

            if options.verbose:
                print(f"Now processing {s}")

            metrics = evaluate(in_dir, options.verbose)
            metrics_agg.append(metrics)
            writer.writerow([s, *metrics])

            if options.verbose:
                print(f"End processing {s}\n")

    out_file = os.path.join(options.out_dir, 'evaluation_metrics_summary.csv')
    with open(out_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        header = ["mean_iou_f1",             "sd_iou_f1", 
                  "mean_abs_lesion_ct_diff", "sd_abs_lesion_ct_diff", 
                  "mean_iou_dice",           "sd_iou_dice", 
                  "mean_dice",               "sd_dice", 
                  "mean_abs_vol_diff",       "sd_abs_vol_diff"]
        writer.writerow(header)
        metrics_row = [np.mean([m[0] for m in metrics_agg]), np.std([m[0] for m in metrics_agg]),
                       np.mean([m[1] for m in metrics_agg]), np.std([m[1] for m in metrics_agg]),
                       np.mean([m[2] for m in metrics_agg]), np.std([m[2] for m in metrics_agg]),
                       np.mean([m[3] for m in metrics_agg]), np.std([m[3] for m in metrics_agg]),
                       np.mean([m[4] for m in metrics_agg]), np.std([m[4] for m in metrics_agg])]
        writer.writerow(metrics_row)

if __name__ == "__main__":
    main()
