# cython: language_level=3
# MIT License

# Copyright (c) 2016-2019 Martin Beroiz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modified by Ignacy Debicki

"""
ASTROALIGN is a simple package that will try to align two stellar astronomical
images, especially when there is no WCS information available.

It does so by finding similar 3-point asterisms (triangles) in both images and
deducing the affine transformation between them.

General registration routines try to match feature points, using corner
detection routines to make the point correspondence.
These generally fail for stellar astronomical images, since stars have very
little stable structure and so, in general, indistinguishable from each other.

Asterism matching is more robust, and closer to the human way of matching
stellar images.

Astroalign can match images of very different field of view, point-spread
functions, seeing and atmospheric conditions.

(c) Martin Beroiz
"""


__version__ = "2.0.1"

__all__ = [
    "MAX_CONTROL_POINTS",
    "MIN_MATCHES_FRACTION",
    "MaxIterError",
    "NUM_NEAREST_NEIGHBORS",
    "PIXEL_TOL",
    "apply_transform",
    "estimate_transform",
    "find_transform",
    "matrix_transform",
    "register",
    "generate_invariants",
]


import math
from collections import Counter
from functools import partial
from itertools import combinations

import numpy as np
import sep
from scipy.linalg import polar
from scipy.spatial import cKDTree

# This should just be pylint being wrong because the file starts with an underscore
# pylint: disable=no-name-in-module
from skimage.transform import matrix_transform  # noqa
from skimage.transform import estimate_transform, warp

MAX_CONTROL_POINTS = 50
"""The maximum control points (stars) to use to build the invariants.

Default: 50"""

PIXEL_TOL = 2
"""The pixel distance tolerance to assume two invariant points are the same.

Default: 2"""

MIN_MATCHES_FRACTION = 0.8
"""The minimum fraction of triangle matches to accept a transformation.

If the minimum fraction yields more than 10 triangles, 10 is used instead.

Default: 0.8
"""

NUM_NEAREST_NEIGHBORS = 5
"""
The number of nearest neighbors of a given star (including itself) to construct
the triangle invariants.

Default: 5
"""


def _invariantfeatures(x1, x2, x3):
    "Given 3 points x1, x2, x3, return the invariant features for the set."
    sides = np.sort(
        [np.linalg.norm(x1 - x2), np.linalg.norm(x2 - x3), np.linalg.norm(x1 - x3)]
    )
    centroid = ((1 / 3) * (x1[0] + x2[0] + x3[0]), (1 / 3) * (x1[1] + x2[1] + x3[1]))
    theta = math.atan2(x1[1] - centroid[1], x1[0] - centroid[0])
    return [
        sides[2] / sides[1],
        sides[1] / sides[0],
        theta,
        x1[0] / 5,
        x1[1] / 5,
        x2[0] / 5,
        x2[1] / 5,
        x3[0] / 5,
        x3[1] / 5,
    ]


def _arrangetriplet(sources, vertex_indices):
    """Return vertex_indices ordered in an (a, b, c) form where:
      a is the vertex defined by L1 & L2
      b is the vertex defined by L2 & L3
      c is the vertex defined by L3 & L1
    and L1 < L2 < L3 are the sides of the triangle defined by vertex_indices."""
    ind1, ind2, ind3 = vertex_indices
    x1, x2, x3 = sources[vertex_indices]

    side_ind = np.array([(ind1, ind2), (ind2, ind3), (ind3, ind1)])
    side_lengths = list(map(np.linalg.norm, (x1 - x2, x2 - x3, x3 - x1)))
    l1_ind, l2_ind, l3_ind = np.argsort(side_lengths)

    # the most common vertex in the list of vertices for two sides is the
    # point at which they meet.

    count = Counter(side_ind[[l1_ind, l2_ind]].flatten())
    a = count.most_common(1)[0][0]
    count = Counter(side_ind[[l2_ind, l3_ind]].flatten())
    b = count.most_common(1)[0][0]
    count = Counter(side_ind[[l3_ind, l1_ind]].flatten())
    c = count.most_common(1)[0][0]

    return np.array([a, b, c])


def colinear_error(p1, p2, p3):
    m = (p1[1] - p2[1]) / (p1[0] - p2[0])
    c = p1[1] - m * p1[0]
    #     if abs(p3[1] - (p3[0] * m + c)) <= 0:
    #         print((p3[1] - (p3[0] * m + c)))
    #         visualize(np.array([p1, p2, p3]), np.array([p1, p2, p3]))
    return abs(p3[1] - (p3[0] * m + c))


def generate_invariants(sources):
    """Return an array of (unique) invariants derived from the array `sources`.
    Return an array of the indices of `sources` that correspond to each invariant,
    arranged as described in _arrangetriplet."""
    arrange = partial(_arrangetriplet, sources=sources)

    inv = []
    triang_vrtx = []
    coordtree = cKDTree(sources)
    # The number of nearest neighbors to request (to work with few sources)
    knn = min(len(sources), NUM_NEAREST_NEIGHBORS)
    for asrc in sources:
        __, indx = coordtree.query(asrc, knn, distance_upper_bound=0.3)

        # Generate all possible triangles with the 5 indx provided, and store
        # them with the order (a, b, c) defined in _arrangetriplet
        all_asterism_triang = [
            arrange(vertex_indices=list(cmb))
            for cmb in combinations(indx, 3)
            if all([idx < len(sources) for idx in cmb])
        ]
        triang_vrtx.extend(all_asterism_triang)

        inv.extend(
            [_invariantfeatures(*sources[triplet]) for triplet in all_asterism_triang]
        )  # if colinear_error(*sources[triplet]) > 0])

    # Remove here all possible duplicate triangles
    uniq_ind = [pos for (pos, elem) in enumerate(inv) if elem not in inv[pos + 1 :]]
    inv_uniq = np.array(inv)[uniq_ind]
    triang_vrtx_uniq = np.array(triang_vrtx)[uniq_ind]

    return inv_uniq, triang_vrtx_uniq


class _MatchTransform:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def fit(self, data):
        """
        Return the best 2D similarity transform from the points given in data.

        data: N sets of similar corresponding triangles.
            3 indices for a triangle in ref
            and the 3 indices for the corresponding triangle in target;
            arranged in a (N, 3, 2) array.
        """
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        approx_t = estimate_transform("similarity", self.source[s], self.target[d])
        return approx_t

    def get_error(self, data, approx_t):
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        resid = approx_t.residuals(self.source[s], self.target[d]).reshape(d1, d2)
        error = resid.max(axis=1)
        return error


def find_transform(
    source,
    target,
    source_invariants=None,
    source_asterisms=None,
    source_invariant_tree=None,
    target_invariants=None,
    target_asterisms=None,
    target_invariant_tree=None,
):
    """Estimate the transform between ``source`` and ``target``.

    Return a SimilarityTransform object ``T`` that maps pixel x, y indices from
    the source image s = (x, y) into the target (destination) image t = (x, y).
    T contains parameters of the tranformation: ``T.rotation``,
    ``T.translation``, ``T.scale``, ``T.params``.

    Args:
        source (array-like): Either a numpy array of the source image to be
            transformed or an interable of (x, y) coordinates of the target
            control points.
        target (array-like): Either a numpy array of the target (destination)
            image or an interable of (x, y) coordinates of the target
            control points.

    Returns:
        The transformation object and a tuple of corresponding star positions
        in source and target.::

            T, (source_pos_array, target_pos_array)

    Raises:
        TypeError: If input type of ``source`` or ``target`` is not supported.
        Exception: If it cannot find more than 3 stars on any input.
    """

    try:
        if len(source[0]) == 2:
            # Assume it's a list of (x, y) pairs
            source_controlp = np.array(source)[:MAX_CONTROL_POINTS]
        else:
            # Assume it's a 2D image
            source_controlp = _find_sources(source)[:MAX_CONTROL_POINTS]
    except TypeError as e:
        raise TypeError("Input type for source not supported.") from e

    try:
        if len(target[0]) == 2:
            # Assume it's a list of (x, y) pairs
            target_controlp = np.array(target)[:MAX_CONTROL_POINTS]
        else:
            # Assume it's a 2D image
            target_controlp = _find_sources(target)[:MAX_CONTROL_POINTS]
    except TypeError as e:
        raise TypeError("Input type for target not supported.") from e

    # Check for low number of reference points
    if len(source_controlp) < 3:
        raise ValueError(
            "Reference stars in source image are less than the minimum value (3)."
        )
    if len(target_controlp) < 3:
        raise ValueError(
            "Reference stars in target image are less than the minimum value (3)."
        )
    source_invariants, source_asterisms = (
        generate_invariants(source_controlp)
        if source_invariants is None or source_asterisms is None
        else (source_invariants, source_asterisms)
    )
    source_invariant_tree = (
        cKDTree(source_invariants)
        if source_invariant_tree is None
        else source_invariant_tree
    )

    target_invariants, target_asterisms = (
        generate_invariants(target_controlp)
        if target_invariants is None or target_asterisms is None
        else (target_invariants, target_asterisms)
    )
    target_invariant_tree = (
        cKDTree(target_invariants)
        if target_invariant_tree is None
        else target_invariant_tree
    )
    # r = 0.1 is the maximum search distance, 0.1 is an empirical value that
    # returns about the same number of matches than inputs
    # matches_list is a list of lists such that for each element
    # source_invariant_tree.data[i], matches_list[i] is a list of the indices
    # of its neighbors in target_invariant_tree.data
    matches = []
    r = 0.1
    i = 0
    # If there is too many matches then reduce the radius of matches
    while (len(matches) == 0 or len(matches) > 15000) and i < 3:
        matches_list = source_invariant_tree.query_ball_tree(target_invariant_tree, r=r)

        # matches unravels the previous list of matches into pairs of source and
        # target control point matches.
        # matches is a (N, 3, 2) array. N sets of similar corresponding triangles.
        # 3 indices for a triangle in ref
        # and the 3 indices for the corresponding triangle in target;
        matches = []
        # t1 is an asterism in source, t2 in target
        for t1, t2_list in zip(source_asterisms, matches_list):
            for t2 in target_asterisms[t2_list]:
                matches.append(list(zip(t1, t2)))
        matches = np.array(matches)
        r /= 2
        i += 1
    inv_model = _MatchTransform(source_controlp, target_controlp)
    n_invariants = len(matches)

    max_iter = n_invariants
    # Set the minimum matches to be between 1 and 10 asterisms
    min_matches = max(1, min(10, int(n_invariants * MIN_MATCHES_FRACTION)))
    try:
        if (len(source_controlp) == 3 or len(target_controlp) == 3) and len(
            matches
        ) == 1:
            best_t = inv_model.fit(matches)
            inlier_ind = np.arange(len(matches))  # All of the indices
        else:
            best_t, inlier_ind = _ransac(
                matches, inv_model, 1, max_iter, PIXEL_TOL, min_matches
            )
    except Exception as e:
        raise e

    triangle_inliers = matches[inlier_ind]
    d1, d2, d3 = triangle_inliers.shape
    inl_arr = triangle_inliers.reshape(d1 * d2, d3)
    inl_unique = set(tuple(pair) for pair in inl_arr)
    inl_arr_unique = np.array(list(list(apair) for apair in inl_unique))
    s, d = inl_arr_unique.T

    return best_t, (source_controlp[s], target_controlp[d]), (s, d)


def apply_transform(transform, source, target, fill_value=None, propagate_mask=False):
    """Applies the transformation ``transform`` to ``source``.

    The output image will have the same shape as ``target``.

    Args:
        transform: A scikit-image ``SimilarityTransform`` object.
        source (numpy array): A 2D numpy array of the source image to be
            transformed.
        target (numpy array): A 2D numpy array of the target image. Only used
            to set the output image shape.
        fill_value (float): A value to fill in the areas of aligned_image
            where footprint == True.
        propagate_mask (bool): Wether to propagate the mask in source.mask
            onto footprint.

    Return:
        A tuple (aligned_image, footprint).
        aligned_image is a numpy 2D array of the transformed source
        footprint is a mask 2D array with True on the regions
        with no pixel information.
    """

    if hasattr(source, "data") and isinstance(source.data, np.ndarray):
        source_data = source.data
    else:
        source_data = source
    if hasattr(target, "data") and isinstance(target.data, np.ndarray):
        target_data = target.data
    else:
        target_data = target

    aligned_image = warp(
        source_data,
        inverse_map=transform.inverse,
        output_shape=target_data.shape,
        order=3,
        mode="constant",
        cval=np.median(source_data),
        clip=False,
        preserve_range=True,
    )
    footprint = warp(
        np.zeros(source_data.shape, dtype="float32"),
        inverse_map=transform.inverse,
        output_shape=target_data.shape,
        cval=1.0,
    )
    footprint = footprint > 0.4

    if hasattr(source, "mask") and propagate_mask:
        source_mask = np.array(source.mask)
        if source_mask.shape == source_data.shape:
            source_mask_rot = warp(
                source_mask.astype("float32"),
                inverse_map=transform.inverse,
                output_shape=target_data.shape,
                cval=1.0,
            )
            source_mask_rot = source_mask_rot > 0.4
            footprint = footprint | source_mask_rot
    if fill_value is not None:
        aligned_image[footprint] = fill_value

    return aligned_image, footprint


def register(source, target, fill_value=None, propagate_mask=False):
    """Transform ``source`` to coincide pixel to pixel with ``target``.

    Args:
        source (numpy array): A 2D numpy array of the source image to be
            transformed.
        target (numpy array): A 2D numpy array of the target image. Only used
            to set the output image shape.
        fill_value (float): A value to fill in the areas of aligned_image
            where footprint == True.
        propagate_mask (bool): Wether to propagate the mask in source.mask
            onto footprint.

    Return:
        A tuple (aligned_image, footprint).
        aligned_image is a numpy 2D array of the transformed source
        footprint is a mask 2D array with True on the regions
        with no pixel information.


    """
    t, _, _ = find_transform(source=source, target=target)
    aligned_image, footprint = apply_transform(
        t,
        source,
        target,
        fill_value,
        propagate_mask,
    )
    return aligned_image, footprint


def _find_sources(img):
    "Return sources (x, y) sorted by brightness."

    if isinstance(img, np.ma.MaskedArray):
        image = img.filled(fill_value=np.median(img)).astype("float32")
    else:
        image = img.astype("float32")
    bkg = sep.Background(image)
    thresh = 3.0 * bkg.globalrms
    sources = sep.extract(image - bkg.back(), thresh)
    sources.sort(order="flux")
    return np.array([[asrc["x"], asrc["y"]] for asrc in sources[::-1]])


# Copyright (c) 2004-2007, Andrew D. Straw. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.

#     * Neither the name of the Andrew D. Straw nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# a PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Modified by Martin Beroiz
# Modified by Ignacy Debicki


class MaxIterError(Exception):
    pass


def _ransac(data, model, min_data_points, max_iter, thresh, min_matches):
    """fit model parameters to data using the RANSAC algorithm

    This implementation written from pseudocode found at
    http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    Given:
        data: a set of data points
        model: a model that can be fitted to data points
        min_data_points: the minimum number of data values required to fit the
            model
        max_iter: the maximum number of iterations allowed in the algorithm
        thresh: a threshold value to determine when a data point fits a model
        min_matches: the min number of matches required to assert that a model
            fits well to data
    Return:
        bestfit: model parameters which best fit the data (or nil if no good model
                  is found)"""
    iterations = 0
    bestfit = None
    best_inlier_idxs = None
    n_data = data.shape[0]
    n = min_data_points
    all_idxs = np.arange(n_data)
    # best_num_inliers = min_matches - 1
    best_fit_closeness = 100
    while iterations < max_iter:
        # Partition indices into two random subsets
        np.random.shuffle(all_idxs)
        maybe_idxs, test_idxs = all_idxs[:n], all_idxs[n:]
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs, :]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < thresh]
        alsoinliers = data[also_idxs, :]
        if len(alsoinliers) >= min_matches:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            current_bestfit = model.fit(betterdata)
            # current_best_num_inliers = len(alsoinliers)
            current_best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
            rotationScaleMatrix = np.copy(current_bestfit.params)
            T = np.copy(rotationScaleMatrix[:2, 2])
            rotationScaleMatrix[0][2] = 0
            rotationScaleMatrix[1][2] = 0
            R, K = polar(rotationScaleMatrix)
            if np.linalg.det(R) < 0:
                R[:2, :2] = -R[:2, :2]
                K[:2, :2] = -K[:2, :2]
            scale = K[0][0]
            rotation = math.acos(R[0][0]) * 180 / math.pi
            fit_closeness = rotation / 60 + abs(scale - 1) / 2 + sum(np.abs(T)) / 2
            # Only terminate if rotation, scale and transformation disturbance are not too far from initial alignment
            if (
                rotation < 7
                and scale > 0.55
                and scale < 1.5
                and sum(np.abs(T)) < 0.2
                and fit_closeness < 0.1
            ):
                bestfit = current_bestfit
                # best_num_inliers = current_best_num_inliers
                best_inlier_idxs = current_best_inlier_idxs
                break
            else:
                # If breaks one of the rules, if it is best match so far anyway, save it
                if fit_closeness < best_fit_closeness:
                    bestfit = current_bestfit
                    # best_num_inliers = current_best_num_inliers
                    best_inlier_idxs = current_best_inlier_idxs
                    best_fit_closeness = fit_closeness
        iterations += 1
    if bestfit is None:
        raise MaxIterError(
            "Max iterations exceeded while trying to find acceptable transformation."
        )

    return bestfit, best_inlier_idxs
