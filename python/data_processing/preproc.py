"""
Preprocessing script for the ISLES24 challenge CTP data
Assume the project directory hierarchy:

prj-dir
|--> derivatives
| |--> sub_stroke####
|   |--> ses-01
|   | |--> sub-stroke####_ses-01_space-ncct_cta.nii.gz
|   | |--> sub-stroke####_ses-01_space-ncct_ctp.nii.gz
|   | |--> perfusion-maps
|   |   |--> sub-stroke####_ses-01_space-ncct_cbf.nii.gz
|   |   |--> sub-stroke####_ses-01_space-ncct_cbv.nii.gz
|   |   |--> sub-stroke####_ses-01_space-ncct_mtt.nii.gz
|   |   |--> sub-stroke####_ses-01_space-ncct_tmax.nii.gz
|   |--> ses-02
|     |--> sub-stroke####_ses-02_lesion-msk.nii.gz
|--> phenotype
| |--> sub_stroke####
|   |--> ses-01
|     |--> sub-stroke####_ses-01_demographic_baseline.csv
|   |--> ses-02
|     |--> sub-stroke####_ses-01_outcome.csv
|--> raw_data
  |--> sub_stroke####
    |--> ses-01
    | |--> sub-stroke####_ses-01_cta.nii.gz
    | |--> sub-stroke####_ses-01_ctp.nii.gz
    | |--> sub-stroke####_ses-01_ncct.nii.gz
    | |--> perfusion-maps
    |   |--> sub-stroke####_ses-01_cbf.nii.gz
    |   |--> sub-stroke####_ses-01_cbv.nii.gz
    |   |--> sub-stroke####_ses-01_mtt.nii.gz
    |   |--> sub-stroke####_ses-01_tmax.nii.gz
    |--> ses-02
      |--> sub-stroke####_ses-02_adc.nii.gz
      |--> sub-stroke####_ses-02_dwi.nii.gz

Where the data in prj-dir/raw_data has been defaced (CT) or skull-stripped (MRI only) and the CTP data has been temporally resampled to 1s and motion corrected via registration to the first timepoint.

The data in prj-dir/derivatives has every subject's imaging registered and resampled to the subject's NCCT, which is located at prj-dir/raw_data/sub_stroke####/ses-01/sub_stroke####_ses-01_ncct.nii.gz

Notably, no methods have been applied to ensure consistency between the different subjects' NCCT spaces; most slice spacing appears to be 2mm, but values of 0.8 and 2.5 are present. The in-slice spacing also differs from 0.3691 to 0.5625, with a mean of 0.4299 and median of 0.4102.

The ground-truth lesion segmentation is located at prj-dir/derivatives/ses-02/sub-stroke####_ses-02_lesion-msk.nii.gz, which is also in the subject's NCCT space and implies, as far as I can tell, that the ground-truths for the evaluation datasets will also have inconsistent in-slice voxel spacing. We will therefore resample the data to a homogenous resolution for machine learning, and then resample the predicted lesions back to NCCT space. A final note - the fact that the ground-truth has been resampled will make the prediction task very difficult. It would be simpler to have everything resampled to the follow-up imaging where the ground-truth lesion was traced, but we will make do with what we have.
"""

import argparse
import os
import os.path as op
import re
import csv
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
        help="Path to the output directory. Files are saved alongside the CTP by default",
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
        "-m",
        "--maps",
        action='store_true',
        help="Also preprocess the perfusion maps, and save them as an additional output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true'
    )

    return parser

def preprocess(ctp_file, seg_file=None, auxiliary_files=None, output_dir='.', verbose=False):
    """Preprocess the ISLES data.

    Parameters
    ----------
    ctp_file : str
        Path to the 4D (x,y,z,t) CTP image to process
    seg_file : str
        Path to the 3D segmentation image to resample using the same parameters and
        then store in the slicewise preprocessed output
    auxiliary_files : tuple(tuple(str, str))
        Paths to additional 3D files to resample using the same parameters and
        then store in the 'auxiliary data' files. Argument should be
        passed in the form ((image_file, output_key), ...)
    output_dir : str
        Path to directory where the processed output should be saved
    verbose : bool
        Print console output
    """

    # Read the images required to process the CTP
    if verbose:
        print(f"Reading CTP image: {ctp_file}")
    raw_img = sitk.ReadImage(ctp_file)

    if seg_file:
        if verbose:
            print(f"Reading SEG image: {seg_file}")
        seg_img = sitk.ReadImage(seg_file)

    ####################################################################
    # Resample such that the size (in-slice) and spacing (in-slice and #
    # through-slice) are consistent between each subject               #
    ####################################################################

    if verbose:
        print("Resampling CTP to ML space")

    # Isolate the brain tissue so that we can identify its bounding box
    baseline_img = (raw_img[..., 0] +
                    raw_img[..., 1] +
                    raw_img[..., 2]) / 3

    # Save out a NIFTI with the world matrix of the NCCT space so that
    # We have a reference to resample the infarct predictions back to
    outfile = op.join(output_dir, 'ctp_baseline_original.nii.gz')
    if verbose:
        print(f"Saving baseline image to {outfile}")
    sitk.WriteImage(baseline_img, outfile)

    # Similarly, the segmentation in NCCT space will be important for
    # evaluation. We don't strictly need to copy the file, but having
    # all of the data extracted to the same directory makes it a lot
    # more convenient for computing and visualizing the results
    if seg_file:
        outfile = op.join(output_dir, 'ctp_seg_original.nii.gz')
        if verbose:
            print(f"Saving segmentation image to {outfile}")
        sitk.WriteImage(seg_img, outfile)

    # Float rounding errors can result in the segmentation having an
    # ever-so-slightly-different physical location to the CTP images,
    # which will cause SimpleITK to have a conniption. We assume that
    # the segmentation is meant to be in the same (NCCT) space as the
    # CTP and copy the imaging information over explicitly
    if seg_file:
        seg_img.CopyInformation(baseline_img)

    # We need to modify the input images so that the axes of the local
    # coordinate system are aligned with, rather than against, the axes of the
    # world coordinte system. This is because SimpleITK's GetArrayFromImage
    # function doesn't look at an image's direction cosines when it is
    # determining the ordering of the image scalars. Two identical images in
    # the same physical space can yield different (mirrored) numpy arrays
    # if their direction cosines have different signs. Fortunately, the
    # FlipImageFilter both reorders the image scalars and modifies the image
    # origin and direction so that the physical location of the object is not
    # changed. That means that we can use it to ensure that our saved numpy
    # arrays have a consistent orientation, regardless of the images' original
    # direction cosines, all the while keeping it aligned with the NCCT space.
    source_direction = np.array(baseline_img.GetDirection()).reshape((3, 3))
    flip_axes = [cos<0 for cos in np.diag(source_direction).tolist()]
    if np.any(flip_axes):
        baseline_img = sitk.Flip(baseline_img, flip_axes)
        helpers.apply_timewise(raw_img, lambda x: sitk.Flip(x, flip_axes))
        # FlipImageFilter is not defined for 4D data and the in-place timewise
        # operation can't change the 4D image's direction cosines. We have to
        # do it manually instead. Such is the price of using a 4D image object.
        baseline_origin = np.array(baseline_img.GetOrigin())
        baseline_direction = np.array(baseline_img.GetDirection()).reshape((3,3))
        raw_origin = np.array(raw_img.GetOrigin())
        raw_direction = np.array(raw_img.GetDirection()).reshape((4,4))
        raw_origin[:3] = baseline_origin
        raw_direction[:3,:3] = baseline_direction
        raw_img.SetOrigin(raw_origin.tolist())
        raw_img.SetDirection(raw_direction.ravel().tolist())
        if seg_file:
            seg_img = sitk.Flip(seg_img, flip_axes)

    # Segment the tissue from the baseline image
    mask = helpers.get_tissue_segmentation_connected_component(baseline_img)

    #######################################################################
    # Resample the tissue to a pre-defined consistent image space for ML. #
    # This operation is functionally a crop, zoom, and interpolation.     #
    #######################################################################

    # The in-slice spacing and extents were chosen empirically such that no
    # tissue would be outside the FOV.
    target_spacing = np.array([0.45, 0.45, 2.0])
    target_slice_extent = np.array([416, 416])

    source_size = np.array(baseline_img.GetSize())
    source_origin = np.array(baseline_img.GetOrigin())
    source_spacing = np.array(baseline_img.GetSpacing())
    source_direction = np.array(baseline_img.GetDirection()).reshape((3, 3))

    if verbose:
        print("Input space:")
        print(f"Size: {source_size}")
        print(f"Origin: {source_origin}")
        print(f"Spacing: {source_spacing}")
        print(f"Direction: {source_direction}")

    # Find the bounding box of the tissue segmentation. Our resampled FOV will
    # center this volume within the target slice extent
    lsif = sitk.LabelStatisticsImageFilter()
    lsif.Execute(baseline_img, mask)
    xmin, xmax, ymin, ymax, zmin, zmax = lsif.GetBoundingBox(1)

    roi_size = np.array((xmax-xmin+1, ymax-ymin+1, zmax-zmin+1))
    roi_output_size = np.ceil(roi_size * source_spacing / target_spacing)
    roi_origin = source_origin + np.matmul(source_direction, np.array((xmin, ymin, zmin)) * source_spacing)

    # Adjust the output size to match the target slice extent, and then the origin to
    # keep the resampled anatomy centered within the image extents
    dx, dy = roi_output_size[:2] - target_slice_extent
    roi_output_size[:2] = target_slice_extent
    roi_origin += np.matmul(source_direction, np.array((dx//2, dy//2, 0)) * source_spacing)

    if verbose:
        print("Output space:")
        print(f"Size: {roi_output_size}")
        print(f"Origin: {roi_origin}")
        print(f"Spacing: {target_spacing}")
        print(f"Direction: {source_direction}")

    if verbose and (dx > 0 or dy > 0):
        print(f'WARNING: Resampling will crop tissue due negative in-slice padding {-dx}, {-dy}')

    # Package the preprocessing steps so far in a neat little method for re-use
    def mask_and_resample(image3D, cast_to=None):
        image3D = sitk.Mask(image3D, mask)
        image3D = sitk.Resample(image3D,
                                roi_output_size.astype(int).tolist(),
                                sitk.Transform(),
                                sitk.sitkLinear,
                                roi_origin.reshape((-1,)).tolist(),
                                target_spacing.tolist(),
                                source_direction.reshape((-1,)).tolist())
        if cast_to:
            image3D = sitk.Cast(image3D, cast_to)
        return image3D

    # Iterate through the timepoints of the 4D CTP and apply the masking and
    # resampling operations. We can't apply the resampling in-place because
    # the image properties change, so new images are made.
    timepoint_imgs = []
    for t in range(raw_img.GetSize()[3]):
        timepoint_img = raw_img[...,t]
        timepoint_img = mask_and_resample(timepoint_img, sitk.sitkFloat32)
        timepoint_imgs.append(timepoint_img)

    # Apply the same preprocessing to the segmentation to keep it aligned
    if seg_file:
        if verbose:
            print("Resampling SEG to ML space")
            # Sanity check to ensure that lesion voxels were not stripped
            lsif.Execute(seg_img, mask)
            stripped_lesion_voxels = lsif.GetSum(0)
            if stripped_lesion_voxels > 0:
                print(f'WARNING: Stripped voxels appearing to be non-tissue were segmented as lesion ({stripped_lesion_voxels} vox)')
        seg_img = mask_and_resample(seg_img, sitk.sitkUInt8)

    # The next step, baseline subtraction, will also require the baseline
    # be aligned to the CTP
    baseline_img = mask_and_resample(baseline_img, sitk.sitkFloat32)

    # Also resample the tissue segmentation to keep it aligned to the CTP
    mask_resampled = mask_and_resample(mask, sitk.sitkUInt8)

    # Save images as NIFTI for quality control, and also to store the 
    # voxel world matrix needed to restore the data to NCCT space when 
    # the model is applied to the testing data
    outfile = op.join(output_dir, 'ctp_baseline_resampled.nii.gz')
    if verbose:
        print(f"Saving resampled baseline to {outfile}")
    sitk.WriteImage(baseline_img, outfile)

    outfile = op.join(output_dir, 'ctp_seg_resampled.nii.gz')
    if verbose:
        print(f"Saving resampled segmentation to {outfile}")
    sitk.WriteImage(seg_img, outfile)

    outfile = op.join(output_dir, 'ctp_mask_resampled.nii.gz')
    if verbose:
        print(f"Saving resampled mask to {outfile}")
    sitk.WriteImage(mask_resampled, outfile)

    ####################################################################
    # Perform baseline subtraction to isolate the contrast signal, and #
    # then use that signal to center a 32-timepoint window on the peak #
    ####################################################################

    # Baseline subtraction
    # Internally, subtracting images with - uses the SubtractImageFilter
    timepoint_imgs = [timepoint_img - baseline_img for timepoint_img in timepoint_imgs]

    # Compute image statistics. The image mean is used for peak detection.
    if verbose:
        print(f"Computing image statistics")

    image_statistics = {}
    for timepoint_img in timepoint_imgs:
        lsif.Execute(timepoint_img, mask_resampled)
        image_statistics.setdefault('minimum',  []).append(lsif.GetMinimum(1))
        image_statistics.setdefault('maximum',  []).append(lsif.GetMaximum(1))
        image_statistics.setdefault('sum',      []).append(lsif.GetSum(1))
        image_statistics.setdefault('variance', []).append(lsif.GetVariance(1))
        image_statistics.setdefault('mean',     []).append(lsif.GetMean(1))
        image_statistics.setdefault('stdev',    []).append(lsif.GetSigma(1))

    # Dump statistics to csv for QC
    outfile = op.join(output_dir, 'image_statistics.csv')
    if verbose:
        print(f"Saving image statistics to {outfile}")
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headings = ['minimum', 'maximum', 'sum', 'variance', 'mean', 'stdev']
        writer.writerow(['timepoint'] + headings)
        for timepoint in range(len(timepoint_imgs)):
            writer.writerow([timepoint] + [image_statistics[heading][timepoint]
                                           for heading
                                           in headings])

    # Compute the 32-timepoint temporal ROI
    if verbose:
        print(f"Computing temporal ROI")
    TEMPORAL_ROI_SIZE = 32
    assert len(timepoint_imgs) >= TEMPORAL_ROI_SIZE
    peak = np.argmax(image_statistics['mean'])
    t_begin = peak - TEMPORAL_ROI_SIZE//2
    t_end = peak + (TEMPORAL_ROI_SIZE-TEMPORAL_ROI_SIZE//2)

    # There are two approaches for 'fixing' situations where the temporal
    # ROI falls outside the min or max indices of the timepoint_imgs:
    # A) Repeat the first or last timepoint to obtain the desired length
    # B) Shift the ROI so that it falls within the bounds of timepoint_imgs
    # In keeping with the previously-written code, we'll implement B
    rshift = 0 - t_begin
    lshift = t_end - len(timepoint_imgs)
    if rshift > 0:
        t_begin += rshift
        t_end += rshift
    if lshift > 0:
        t_begin -= lshift
        t_end -= lshift

    # An argument could also be made for computing the temporal ROI as the
    # 32 timepoints following the onset of the contrast, to try and preserve
    # differences in peak enhancement time between patients. Onset here could
    # be calculated by taking the baseline enhancement (mean over the first
    # three timepoints) and multiplying by 1.15 to obtain an onset threshold.

    if verbose:
        print(f"Peak enhancement at timepoint {peak}/{len(timepoint_imgs)-1} yields ROI [{t_begin}:{t_end}]")

    ####################################################################
    # Dump the image scalars to a numpy array so that we can perform   #
    # z-score normalization and then save the scalar data for training #
    ####################################################################

    # Stack timepoint image scalars to obtain x, y, z, t ordering
    # The transposition and flip may look weird, but the data ordering
    # used by SimpleITK and numpy differ such that it is necessary
    ctp_array = np.stack([helpers.image_to_ndarr(timepoint_img)
                          for timepoint_img
                          in timepoint_imgs[t_begin : t_end]], axis=-1)
    mask_array = helpers.image_to_ndarr(mask_resampled)==1
    
    # Z-score normalization per patient -- normalization is being done across all slices and time points
    ctp_array = helpers.z_score_normalization(ctp_array, np.repeat(mask_array[..., np.newaxis],
                                                                   TEMPORAL_ROI_SIZE,
                                                                   axis=-1))

    # If a segmentation was given, process it to match the ctp array
    if seg_file:
        seg_array = helpers.image_to_ndarr(seg_img)

    # Save the preprocessed CTP data. Each file represents a single slice
    # across all 32 timepoints (img) and its grounds truth (label)
    if verbose:
        print(f"Saving slicewise image scalars to {output_dir}/ctp_preprocessed_slice_####.npz")
    for slc_idx in range(ctp_array.shape[2]):
        outfile = op.join(output_dir, f'ctp_preprocessed_slice_{slc_idx}.npz')
        outdata = {'img' : ctp_array[:, :, slc_idx, :],
                   'mask' : mask_array[:, :, slc_idx, np.newaxis]}
        if seg_file:
            outdata['label'] = seg_array[:, :, slc_idx, np.newaxis]
        np.savez_compressed(outfile, **outdata)
        
    if auxiliary_files:
        if verbose:
            print(f"Processing auxiliary images and saving scalars to {output_dir}/ctp_auxiliary_slice_####.npz")
        auxiliary_data = {}
        for auxiliary_file, output_key in auxiliary_files:
            auxiliary_img = sitk.ReadImage(auxiliary_file)
            if np.any(flip_axes):
                auxiliary_img = sitk.Flip(auxiliary_img, flip_axes)
            auxiliary_img = mask_and_resample(auxiliary_img, sitk.sitkFloat32)
            auxiliary_data[output_key] = sitk.GetArrayFromImage(auxiliary_img).transpose([2,1,0])
        for slc_idx in range(ctp_array.shape[2]):
            outfile = op.join(output_dir, f'ctp_auxiliary_slice_{slc_idx}.npz')
            np.savez_compressed(outfile, **{output_key: auxiliary_data[output_key][:, :, slc_idx]
                                            for output_key
                                            in auxiliary_data})

def main(args=None):
    """entry point."""

    options = _get_parser().parse_args(args)
    # Parse options to determine which subject IDs to process
    if options.batch:
        # Discover the subject data, unless subject IDs were also provided, which would be invalid.
        if options.sbj_id:
            print('ERROR: Only use one of -s and -b')
            exit()
        # Instead of searching all of prj_dir, we'll assume the directory structure given at the top
        derivatives_dir = os.path.join(options.prj_dir, 'derivatives')
        # For interactive explanations of the regex patterns, see https://regex101.com/
        sbj_dir_pattern = r"^sub-stroke(\d{4})$"
        sbj_dirs = os.listdir(derivatives_dir)
        options.sbj_id = []
        for sbj_dir in sbj_dirs:
            match = re.match(sbj_dir_pattern, sbj_dir)
            if match:
                options.sbj_id.append(match[1])
        if options.verbose:
            print(f"Discovered the following subject directories in {derivatives_dir}:")
            for s in options.sbj_id:
                print(s)
    # Require one of -s or -b. Otherwise, there are no sbj_id to process
    elif not options.sbj_id:
        print('ERROR: Use one of -s or -b')
        exit()

    # Process each subject ID:
    for s in options.sbj_id:
        ctp_file = os.path.join(options.prj_dir,
                                "derivatives",
                                f"sub-stroke{s}",
                                "ses-01",
                                f"sub-stroke{s}_ses-01_space-ncct_ctp.nii.gz")
        seg_file = os.path.join(options.prj_dir,
                                "derivatives",
                                f"sub-stroke{s}",
                                "ses-02",
                                f"sub-stroke{s}_ses-02_lesion-msk.nii.gz")

        maps = None
        if options.maps:
            cbf_file = os.path.join(options.prj_dir,
                                    "derivatives",
                                    f"sub-stroke{s}",
                                    "ses-01",
                                    "perfusion-maps",
                                    f"sub-stroke{s}_ses-01_space-ncct_cbf.nii.gz")
            cbv_file = os.path.join(options.prj_dir,
                                    "derivatives",
                                    f"sub-stroke{s}",
                                    "ses-01",
                                    "perfusion-maps",
                                    f"sub-stroke{s}_ses-01_space-ncct_cbv.nii.gz")
            mtt_file = os.path.join(options.prj_dir,
                                    "derivatives",
                                    f"sub-stroke{s}",
                                    "ses-01",
                                    "perfusion-maps",
                                    f"sub-stroke{s}_ses-01_space-ncct_mtt.nii.gz")
            tmax_file = os.path.join(options.prj_dir,
                                     "derivatives",
                                     f"sub-stroke{s}",
                                     "ses-01",
                                     "perfusion-maps",
                                     f"sub-stroke{s}_ses-01_space-ncct_tmax.nii.gz")
            maps = ((cbf_file,  'cbf'),
                    (cbv_file,  'cbv'),
                    (mtt_file,  'mtt'),
                    (tmax_file, 'tmax'))
        if options.out_dir:
            out_dir = os.path.join(options.out_dir, s)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
        else:
            out_dir = os.path.join(options.prj_dir,
                                   "derivatives",
                                   f"sub-stroke{s}",
                                   "ses-01")

        if options.verbose:
            print(f"Now processing {s}")

        preprocess(ctp_file,
                   seg_file,
                   maps,
                   out_dir,
                   options.verbose)

        if options.verbose:
            print(f"End processing {s}\n")

if __name__ == "__main__":
    main()
