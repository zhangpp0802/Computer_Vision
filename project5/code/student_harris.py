import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(img, feature_width):
    """

    JR adds: to ensure compatability with project 4A, you simply need to use
    this function as a wrapper for your 4A code.  Guidelines below left
    for historical reference purposes.

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
    k = 0.04
    # get the shape of the image
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = np.delete(img, 1, 2)
    img = np.delete(img, 1, 2)
    print(img.shape)
    m, n, c = img.shape
    img = img[:,:,0]    # here, only get one layer of colors, as we assume grayscale

    # calculate the x derivative and the y derivative with Sobel
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

    # compute each matrix
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # convolve each image
    gaussian_kernel = cv2.getGaussianKernel(ksize=5, sigma=1)
    Ixx = cv2.filter2D(Ixx, cv2.CV_64F, gaussian_kernel)
    Ixy = cv2.filter2D(Ixy, cv2.CV_64F, gaussian_kernel)
    Iyy = cv2.filter2D(Iyy, cv2.CV_64F, gaussian_kernel)

    # compute R strength
    R = (Ixx*Iyy-Ixy**2) - k*(Ixx+Iyy)**2

    # threshold R with 1% of maximum value in R
    R[R<0.01*np.amax(R)] = 0
    Rvals = np.copy(R)

    # construct 2d-array "R" filled with arrays containing x, y, r-value
    R = []
    m, n = Rvals.shape
    Rmax = np.amax(Rvals)
    for y in range(0, m):
        for x in range(0, n):
            if not Rvals[y, x] == 0:
                R.append([int(x), int(y), 255 * Rvals[y, x] / Rmax])

    # sort R by Rvals in descending order
    R = np.asarray(R)
    R = R[R[:, 2].argsort()]
    R = np.flipud(R)

    numPts = 2000
    radiis = []

    # go through Rvals
    for i, thisPoint in enumerate(R):
        if i == 0:
            radius = 9999999.0  # arbitrary large number
            thisPtSet = (thisPoint[0], thisPoint[1], radius)
            radiis.append(thisPtSet)
            continue

        # these are key R to be compared with "this point"
        otherPoints = R[0:i, :]

        # compute distance between the current point and another point
        distances = np.sqrt(((otherPoints[:, 0] - thisPoint[0]) ** 2 + (otherPoints[:, 1] - thisPoint[1]) ** 2))

        # get the minimum distance
        min_dist = np.min(distances)

        # get the index of the keypoint with min radii
        thisPtSet = (thisPoint[0], thisPoint[1], min_dist)
        radiis.append(thisPtSet)

    # sort radiis in descending order
    radiis = np.asarray(radiis)
    radiis = radiis[radiis[:, 2].argsort()]
    radiis = np.flipud(radiis)

    radiis = radiis[0:numPts, :]

    y = radiis[:, 1]
    x = radiis[:, 0]

    return x, y, confidences, scales, orientations#, scales, orientations
