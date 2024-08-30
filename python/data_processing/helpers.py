import SimpleITK as sitk
import numpy as np

# Documenting as per https://numpydoc.readthedocs.io/en/latest/format.html

def apply_timewise(image4D, func3D):
    """ Transform each timepoint of a 4D SimpleITK image in-place.

        The function func3D is applied seperately to each timepoint in
        image4D and its return value is stored back into image4D.

        Parameters
        ----------
        image4D : SimpleITK.Image
            Spatiotemporal data with (x,y,z,t) axis ordering.
        func3D : Callable[SimpleITK.Image, SimpleITK.Image]
            The method to apply to each timepoint. The size, dimensions
            and scalar type of the returned image must match the input image.

    """
    for t in range(image4D.GetSize()[3]):
        # Internally, indexing a SimpleITK.Image object uses
        # ExtractImageFilter and PasteImageFilter, so this ends up just
        # being shorthand for the canonical approach seen in the C++ examples
        image4D[...,t] = func3D(image4D[..., t])

def apply_slicewise(image3D, func2D):
    """ Transform each axial slice of a 3D SimpleITK image in-place.

        The function func2D is applied seperately to each axial slice in
        image3D and its return value is stored back into image3D.

        Parameters
        ----------
        image3D : SimpleITK.Image
            Spatial data with (x,y,z) axis ordering.
        func2D : Callable[SimpleITK.Image, SimpleITK.Image]
            The method to apply to each axial slice. The size, dimensions
            and scalar type of the returned image must match the input image.

    """

    for z in range(image3D.GetSize()[2]):
        image3D[...,z] = func2D(image3D[..., z])

def get_csf_segmentation(image3D):
    image3D = sitk.DiscreteGaussian(image3D, (1.,1.,0.))
    return sitk.BinaryThreshold(image3D, -1, 15)

def get_csf_segmentation_adaptive(image3D, tissue_segmentation):
    """ Obtain a binary mask denoting the cerebrospinal fluid of an NCCT-like image.

        The input image is assumed to be a 3D image with (x,y,z) axis ordering
        with a field of view covering the head and neck. A tissue segmentation
        is necessary to derive the threshold for identifying the CSF, and can be
        obtained from any of the get_tissue_segmentation_* functions

        Parameters
        ----------
        image3D : SimpleITK.Image
            NCCT-like head image to segment the CSF from
        tissue_segmentation : SimpleITK.Image
            Binary mask covering the brain tissue and CSF

        Returns
        -------
        SimpleITK.Image
            CSF segmentation in the coordinate space of the input image.
    """
    lsif = sitk.LabelStatisticsImageFilter()
    lsif.Execute(image3D, tissue_segmentation)
    tissue_voxels = lsif.GetCount(1)
    for threshold in range(5, int(lsif.GetMedian(1))):
        seg = sitk.BinaryThreshold(image3D, -1, threshold)
        seg = sitk.Mask(seg, tissue_segmentation)
        apply_slicewise(seg, lambda x: sitk.BinaryDilate(x, (3,3)))
        ccs = sitk.ConnectedComponent(seg)
        ccs = sitk.RelabelComponent(ccs)
        lsif.Execute(image3D, ccs)
        cc_voxels = lsif.GetCount(1)
        if cc_voxels/tissue_voxels >= 0.025:
            break
    image3D = sitk.DiscreteGaussian(image3D, (1.,1.,0.))
    mask = sitk.BinaryThreshold(image3D, -1.0, threshold)
    mask = sitk.Mask(mask, tissue_segmentation)
    return mask

def get_tissue_segmentation_connected_component(image3D):
    """ Obtain a binary mask denoting the brain tissue of an NCCT-like image.

        The input image is assumed to be a 3D image with (x,y,z) axis ordering
        with a field of view covering the head and neck. A field of view that
        also covers the upper torso will likely fail. The tissue segmentation
        is computed via the following steps:
        1. Binary thresholding and morphological opening to segment tissue
        2. Remove all but the largest connected component to isolate brain
           ** NOTE THIS IS THE MOST LIKELY POINT OF FAILURE.
        3. Reapply tissue mask and apply in-slice hole filling to clean up
           the edges and fill in the ventricles

        Parameters
        ----------
        image3D : SimpleITK.Image
            NCCT-like head image to segment the brain tissue from.

        Returns
        -------
        SimpleITK.Image
            Tissue segmentation in the coordinate space of the input image.
    """
    # Small gaussian blur reduces noise to improve thresholding
    image3D = sitk.DiscreteGaussian(image3D, (1.,1.,0.))
    # Brain tissue typically falls in [20, 45] HU and skull in [500, 1900]
    # In practice [1, 100] delineates tissue well from skull (101+)
    thresholded = sitk.BinaryThreshold(image3D, 1.0, 100.0)
    # Separate parts connected by only a few voxels
    mask = sitk.BinaryErode(thresholded, (5,5,1))
    # Keep largest connected component
    mask = sitk.ConnectedComponent(mask)
    mask = sitk.RelabelComponent(mask)
    mask = sitk.BinaryThreshold(mask, 1, 1)
    # Restore boundary of image that was previously eroded with a small pad
    mask = sitk.BinaryDilate(mask, (7,7,1))
    # Restore the boundary of the thresholded image (dilation is 'blobby')
    mask = sitk.Mask(mask, thresholded)
    # Slicewise hole filling (eg. for ventricles)
    apply_slicewise(mask, lambda x: sitk.BinaryFillhole(x))
    return mask

def get_tissue_segmentation_region_growing(image3D):
    """ Obtain a binary mask denoting the brain tissue of an NCCT-like image.

        The input image is assumed to be a 3D image with (x,y,z) axis ordering
        with a field of view covering the head and neck. A field of view that
        also covers the upper torso will likely be tolerated. The image
        scalars must be recorded in Hounsfield units. The tissue segmentation
        is computed via the following steps:
        1. Identify the slice of the image containing the most brain tissue
           fully-enclosed in skull. This identifies the head from the torso.
           ** NOTE THAT THIS CAN FAIL IF THE SKULL EXTENDS PAST THE EDGE
              OF THE IMAGE ON EVERY SLICE (THERE IS NO ENCLOSED BRAIN).S
        2. Seed a region-growing approach with voxels from 1. that are
           inside the skull and have the intensity of brain tissue.
        3. Grow the region outward, stopping at skull tissue or when the
           cross-sectional area of the brain tissue below the skull is small
           enough to suggest that we are at the brainstem and about to enter
           the upper torso. Segmenting the brain slicewise out from the center
           guarantees that the region growing will not leak around the bottom
           of the skull and rise back up the soft tissue outside the skull.

        Parameters
        ----------
        image3D : SimpleITK.Image
            NCCT-like head image to segment the brain tissue from.

        Returns
        -------
        SimpleITK.Image
            Tissue segmentation in the coordinate space of the input image.
    """

    # Small gaussian blur reduces noise to improve thresholding
    image3D = sitk.DiscreteGaussian(image3D, (1.,1.,0.))
    # Brain tissue typically falls in [20, 45] HU and skull in [500, 1900]
    # In practice [1, 100] delineates tissue well from skull (101+)
    skull_seg = sitk.BinaryThreshold(image3D, 101, 32767.)
    tissue_seg = sitk.BinaryThreshold(image3D, 1, 100)
    # Erosion to remove flecks of soft tissue and sever the optic nerve
    tissue_seg = sitk.BinaryErode(tissue_seg, (5,5,1))

    # Find the slice with fully-enclosed skull containing the most brain
    max_area_slc = {'z_index':0, 'size':0, 'seeds':None}
    lssif = sitk.LabelShapeStatisticsImageFilter()
    z_size = skull_seg.GetSize()[-1]
    for zidx in range(z_size):
        slc = skull_seg[:,:,zidx]
        flc = sitk.BinaryFillhole(slc)
        slc = sitk.MaskNegated(flc, slc)
        slc = sitk.Mask(tissue_seg[:,:,zidx], slc)
        lssif.Execute(slc)
        if lssif.GetNumberOfLabels() < 1:
            continue
        _size = lssif.GetNumberOfPixels(1)
        if _size > max_area_slc['size']:
            max_area_slc['size'] = _size
            max_area_slc['z_index'] = zidx
            max_area_slc['seeds'] = slc

    if max_area_slc['seeds'] is None:
        zidx = z_size//2
        slc = sitk.MaskNegated(tissue_seg[:,:,zidx], skull_seg[:,:,zidx])
        lssif.Execute(slc)
        if lssif.GetNumberOfLabels() < 1:
            print('Could not seed brain region. Falling back to CCA.')
            return get_tissue_segmentation_connected_component(image3D)
        max_area_slc = {'z_index':zidx, 'size':lssif.GetNumberOfPixels(1), 'seeds':slc}

    # Flood the tissue segmentation out from the enclosed region
    # identified above. Flooding only up/down prevents most 'leaks'.
    def get_seeds_from_segmentation(seg):
        arr = sitk.GetArrayFromImage(seg)
        idxs = np.transpose(np.nonzero(arr))
        return [(int(x),int(y)) for y,x in idxs]

    slc_seeds = get_seeds_from_segmentation(max_area_slc['seeds'])
    slc = sitk.ConnectedThreshold(tissue_seg[:,:,max_area_slc['z_index']],
                                  seedList=slc_seeds,
                                  lower=1,
                                  upper=1)
    slc = sitk.BinaryFillhole(slc)
    tissue_seg[:,:,max_area_slc['z_index']] = slc
    central_seeds = get_seeds_from_segmentation(slc)

    # Flood up
    slc_seeds = central_seeds
    for zidx in range(max_area_slc['z_index']+1, z_size):
        if len(slc_seeds) < 1:
            tissue_seg[:,:,zidx:z_size] = 0
            break
        slc = tissue_seg[:,:,zidx]
        slc = sitk.ConnectedThreshold(slc, seedList=slc_seeds, lower=1, upper=1)
        tissue_seg[:,:,zidx] = sitk.BinaryFillhole(slc)
        slc_seeds = get_seeds_from_segmentation(slc)

    # Flood down
    slc_seeds = central_seeds
    last_mask = None
    for zidx in range(max_area_slc['z_index']-1, -1, -1):
        slc = tissue_seg[:,:,zidx]
        slc = sitk.ConnectedThreshold(slc, seedList=slc_seeds, lower=1, upper=1)

        # Extra logic only for flooding down, to prevent the flood from leaking
        # out of the skull and into the neck, body
        cc = sitk.ConnectedComponent(slc, True)
        cc = sitk.RelabelComponent(cc)
        lssif.Execute(cc)
        if lssif.GetNumberOfLabels() < 1:
            tissue_seg[:,:,0:zidx+1] = 0
            break
        first_size = lssif.GetNumberOfPixels(1)
        vox_x, vox_y, _ = image3D.GetSpacing()
        if first_size * vox_x * vox_y < 4750:
            if last_mask:
                last_mask.SetOrigin(slc.GetOrigin())
                slc = sitk.Mask(slc, last_mask)
            last_mask = slc
            if lssif.GetNumberOfLabels() > 1:
                second_size = lssif.GetNumberOfPixels(2)
                if first_size/second_size < 2:
                    tissue_seg[:,:,0:zidx+1] = 0
                    break

        tissue_seg[:,:,zidx] = sitk.BinaryFillhole(slc)
        slc_seeds = get_seeds_from_segmentation(slc)

    # Restore the boundary of the brain that was previously eroded
    return sitk.BinaryDilate(tissue_seg, (7,7,1))

def image_to_ndarr(image3D):
    """ Convert a SimpleITK.image to a numpy.ndarray.

        Given a 3D SimpleITK image, obtain a numpy ndarr such that
        indexing both image representations with the same indices will
        return the scalars for identical regions of the image. Some
        processing is necessary beyond SimpleITK's GetArrayFromImage
        method because SimpleITK and numpy assume different axis ordering
        when iterating over multidimensional arrays; the axis order is
        reversed if the scalar data is simply copied between modules.

        Parameters
        ----------
        image3D : SimpleITK.Image
            The image object to extract the scalar data from.

        Returns
        -------
        numpy.ndarr
            The extracted scalar data as a ndarr object.
    """

    return sitk.GetArrayFromImage(image3D).transpose([2,1,0])

def ndarr_to_image(ndarr3D):
    """ Convert a numpy.ndarray to a SimpleITK.image.

        Inverse operation of image_to_ndarr.

        Parameters
        ----------
        ndarr3D : numpy.ndarr
            The array object to extract the scalar data from.

        Returns
        -------
        numpy.ndarr
            The extracted scalar data as an Image object.
    """

    return sitk.GetImageFromArray(ndarr3D.transpose([2,1,0]))

def orient_axial_slice(ndarr2D):
    """ Orient data for matplotlib.pyplot.imshow per radiological convention

        Given a 3D ndarr with (x,y,z) axis order from image_to_ndarr,
        an axial slice (taken through the Z axis) will not render in
        proper radiological orientation if passed directly to
        matplotlib.pyplot.imshow. This method manipulates the axes of
        a 2D axial slice such that the patient's anterior is shown at
        the top of the screen, posterior at the bottom, right on the
        left, and left on the right. It is as if the patient is lying
        on their back and you are looking through the bottom of their
        feet, with lower-index slices being closer to their toes and
        higher index slices being closer to the top of their head.

        Parameters
        ----------
        ndarr2D : numpy.ndarr
            Scalar data of the axial slice to be reoriented.

        Returns
        -------
        numpy.ndarr
            The reoriented scalar data
    """
    if ndarr2D.ndim>2:
        ndarr2D = np.squeeze(ndarr2D)
    return ndarr2D.transpose()

def z_score_normalization(ndarr, mask=None):
    """ Compute the Z score of every voxel in an image with respect to a ROI

        Compute the mean and standard deviation of a ROI, then use those
        parameters to transform every element of the input to its Z score

        Parameters
        ----------
        ndarr : numpy.ndarr
            Scalar data to be normalized
        mask : numpy.ndarr, optional
            Binary mask representing the ROI

        Returns
        -------
        numpy.ndarr
            The normalized scalar data
    """

    ma = np.ma.array(ndarr, mask=np.ma.nomask if mask is None else np.invert(mask))
    return (ndarr-ma.mean())/ma.std()

