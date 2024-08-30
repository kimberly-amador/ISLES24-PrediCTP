"""
Postprocessing script for the ISLES24 challenge model output
Assume the project directory hierarchy:

prj-dir
|--> ####
  |--> model_prediction.npz

Where the file model_prediction.npz contains an object with the key 'pred'
that evaluates to a numpy array of size (slices, 416, 416, 2). The last
element of the inner dimension should be the voxelwise infarct probabilities
which fall in the range [0, 1]
"""

import argparse
import os
import os.path as op
import re
import numpy as np
import SimpleITK as sitk
# Local imports:
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

def postprocess(in_dir, out_file, verbose):
    # Get a 3D numpy array of image scalars somehow
    in_path = os.path.join(in_dir, 'model_prediction.npz')
    if verbose:
        print(f'Reading model prediction from {in_path}')
    image_scalars = np.load(in_path)['pred']
    image_scalars = image_scalars[..., 1] # Extract lesion label
    image_scalars = np.moveaxis(image_scalars, 0, -1) # ZXY > XYZ

    # Threshold to binary segmentation
    image_scalars = np.rint(image_scalars).astype(np.uint8)

    # Morphological processing to clean up lesion boundary
    segmentation_image = helpers.ndarr_to_image(image_scalars)
    segmentation_image = sitk.BinaryMorphologicalOpening(segmentation_image, [1,1,0])
    segmentation_image = sitk.BinaryDilate(segmentation_image, [2,2,0])

    # Restore world information to put seg in ML space
    information_image = sitk.ReadImage(os.path.join(in_dir, 'ctp_baseline_resampled.nii.gz'))
    segmentation_image.CopyInformation(information_image)

    # Resample from ML space to NCCT space
    reference_image = sitk.ReadImage(os.path.join(in_dir, 'ctp_baseline_original.nii.gz'))
    segmentation_image = sitk.Resample(segmentation_image, reference_image)

    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if verbose:
        print(f'Writing lesion segmentation to {out_file}')
    sitk.WriteImage(segmentation_image, out_file)

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

        out_file = os.path.join(out_dir, f'lesion_segmentation.nii.gz')

        if options.verbose:
            print(f"Now processing {s}")

        postprocess(in_dir,
                    out_file,
                    options.verbose)

        if options.verbose:
            print(f"End processing {s}\n")

if __name__ == "__main__":
    main()
