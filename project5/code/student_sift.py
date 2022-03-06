import numpy as np
import cv2
import math


def get_features(image, x, y, feature_width, scales=None):
    """
    JR Writes: To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    maximal points you may need to implement a more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)


    Below for advanced implementation:

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################

    # get sobel filters
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # compute angles and magnitudes of the image
    angles = np.arctan2(sobelY, sobelX)
    magnitudes = np.sqrt(sobelX ** 2 + sobelY ** 2)

    fv = []

    # go through each key point
    for onePoint, val in enumerate(x):
        x1 = int(x[onePoint])
        y1 = int(y[onePoint])

        # get patch
        anglesPatch = angles[y1 - 8:y1 + 8, x1 - 8:x1 + 8]
        magnitudesPatch = magnitudes[y1 - 8:y1 + 8, x1 - 8:x1 + 8]
        siftFeatures = []

        # use patch for each key point
        for i in range(0, 16, 4):
            for j in range(0, 16, 4):
                # get bins for each patch
                anglesPatchBin = anglesPatch[i:i + 4, j:j + 4]
                magnitudesPatchBin = magnitudesPatch[i:i + 4, j:j + 4]

                # compute histogram and normalize
                hist, bins = np.histogram(anglesPatchBin, bins=8, range=(-math.pi, math.pi), weights=magnitudesPatchBin)
                siftFeatures.extend(hist)

        siftFeatures = np.array(siftFeatures)

        fv.append(siftFeatures)

    fv = np.array(fv)
    fv = fv**.7
    return fv
