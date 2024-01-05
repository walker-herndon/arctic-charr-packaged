import json
import os
import pickle

import cv2
import numpy as np
import skimage.transform
from scipy.spatial import cKDTree

from . import astroalign
from .util import (
    crop_image,
    get_average_precision_recall,
    get_normalise_direction_matrix,
    visualize,
)

astroalign.MAX_CONTROL_POINTS = 50
astroalign.MIN_MATCHES_FRACTION = 0.8
astroalign.NUM_NEAREST_NEIGHBORS = 15
astroalign.PIXEL_TOL = 0.01

__cache_dir__ = "aa_cache"


def aamatch(
    spots_1,
    spots_2,
    source_invariants=None,
    source_asterisms=None,
    source_invariant_tree=None,
    target_invariants=None,
    target_asterisms=None,
    target_invariant_tree=None,
):
    """Attempts to match 2 lists of spots.

    Args:
        spots_1 (List[List[float]]): The list of spots in list 1
        spots_2 (List[List[float]]): The list of spots in list 2
        source_invariants (Object, optional): The cached triangle invariants for list 1 if available. Defaults to None.
        source_asterisms (Object, optional): The triangle asterisms for list 1 if available. Defaults to None.
        source_invariant_tree (Object, optional): The invariant kd-tree for list 1 if available. Defaults to None.
        target_invariants (Object, optional): The cached triangle invariants for list 2 if available. Defaults to None.
        target_asterisms (Object, optional): The triangle asterisms for list 2 if available. Defaults to None.
        target_invariant_tree (Object, optional): The invariant kd-tree for list 2 if available. Defaults to None.

    Returns:
        Tuple[np.ndarray, int, float, List[int], List[int]]: Tuple of the Affine transform matrix mapping one point onto the other, the number of inliers, the score and the matching indexes form list 1 and list 2
    """
    try:
        T, (s_pos, t_pos), (s_idx, t_idx) = astroalign.find_transform(
            spots_1,
            spots_2,
            source_invariants=source_invariants,
            source_asterisms=source_asterisms,
            source_invariant_tree=source_invariant_tree,
            target_invariants=target_invariants,
            target_asterisms=target_asterisms,
            target_invariant_tree=target_invariant_tree,
        )
        return (
            T,
            len(s_pos),
            (len(s_pos) / len(spots_1)) * (len(t_pos) / len(spots_2)),
            s_idx,
            t_idx,
        )
    except TypeError:
        return None, 0, 0, [], []
    except ValueError:
        return None, 0, 0, [], []
    except astroalign.MaxIterError:
        return None, 0, 0, [], []


def retrieve_from_cache(imgKey, cache_dir):
    """Retrieves the precomputed values from cache if present

    Args:
        imgKey (str): The key for which to retrieve the precomputed values

    Returns:
        Dict|None: The object stored in the cache for the key
    """
    if os.path.isfile(os.path.join(cache_dir, imgKey + ".aa.pickle")):
        with open(os.path.join(cache_dir, imgKey + ".aa.pickle"), "rb") as f:
            precomputedModelValues = pickle.load(f)
        return precomputedModelValues
    else:
        return None


def open_cached(
    fish,
    cache_dir="aa_cache",
):
    """Attempts to open a cached version of precomputed values for the comparison. Computes these if necessary and saves to cache.

    Args:
        fish (Fish): The fish object containing the information for comparison.
        cache_dir (str, optional): The directory where the cached values are stored. Defaults to "aa_cache".

    Returns:
        dict: The dictionary of precomputed values.
    """
    precomputedModelValues = None
    if fish.precompAA is not None:
        with open(fish.precompAA, "rb") as f:
            precomputedModelValues = pickle.load(f)
    else:
        precomputedModelValues = retrieve_from_cache(fish.uuid, cache_dir)

    if (
        precomputedModelValues is None
        or "version" not in precomputedModelValues
        or precomputedModelValues["version"] != 1
        or precomputedModelValues["nn"] != astroalign.NUM_NEAREST_NEIGHBORS
        or precomputedModelValues["max_control_points"] != astroalign.MAX_CONTROL_POINTS
    ):
        precomputedModelValues = precomputeValues(
            fish,
            cache_dir=cache_dir,
        )

    return precomputedModelValues


def precomputeValues(
    fish,
    cache_dir="aa_cache",
):
    """Precomputes values necessary for comparison of the fish to another fish

    Args:
        fish (Fish): The fish object for which to precompute values.
        cache_dir (str, optional): The directory where the precomputed values will be cached. Defaults to "aa_cache".

    Returns:
        dict: The dictionary of precomputed values.
    """
    spots = None
    if fish.spotJson is not None:
        with open(fish.spotJson, "r", encoding="utf-8") as f:
            spots = json.load(f)
        if len(spots) > 5:
            spots = [n[:2] for n in spots]  # Remove size from spots
            spots = [
                spots[i]
                for i in range(len(spots))
                if i == 0 or spots[i] != spots[i - 1]
            ]  # Remove duplicates
            mask = crop_image(cv2.imread(fish.mask_path, 0))
            R = get_normalise_direction_matrix(mask)
            tmpPoints = np.copy(np.asarray(spots))
            tmpPoints = np.dot(
                R[:, (0, 1)], np.array([tmpPoints[:, 0], tmpPoints[:, 1]])
            )
            tmpPoints = np.array(list(zip(tmpPoints[0], tmpPoints[1])))
            warpedTargetMask = crop_image(
                cv2.warpAffine(mask, R, (mask.shape[1] + 500, mask.shape[0] + 500))
            )
            spotsProcessed = tmpPoints / np.max(warpedTargetMask.nonzero())

            invariants, asterisms = astroalign.generate_invariants(
                spotsProcessed[: astroalign.MAX_CONTROL_POINTS]
            )
            if np.isnan(invariants).any() or np.isinf(invariants).any():
                return None
            kdTree = cKDTree(invariants)
            fish.precompAA = os.path.join(cache_dir, fish.uuid + ".aa.pickle")
            precomputedObject = {
                "max_control_points": astroalign.MAX_CONTROL_POINTS,
                "invariants": invariants,
                "asterisms": asterisms,
                "kdtree": kdTree,
                "spots": spots,
                "spots_standardised": spotsProcessed,
                "nn": astroalign.NUM_NEAREST_NEIGHBORS,
                "version": 1,
            }
            with open(os.path.join(cache_dir, fish.uuid + ".aa.pickle"), "wb") as f:
                pickle.dump(precomputedObject, f)
            return precomputedObject
    return None


def findClosestMatch(
    query_fish,
    fishToComare,
    cache_dir="aa_cache",
    verbose=False,
    progress=False,
):
    """Uses the Astroalign matching algorithm to find the best matches for the model image in the images to compare dictionaries

    Args:
        query_fish (Fish): The model fish image to compare.
        fishToComare (List[Fish]): The list of fish images to compare to.
        cache_dir (str, optional): The directory to store cached precomputed values. Defaults to "aa_cache".
        verbose (bool, optional): If True, print verbose information to stdout. Defaults to False.
        progress (bool, optional): If True, print progress information to stdout. Defaults to False.

    Returns:
        List[Tuple[float, float, str]]: List of tuples of (score, maskCoverage, image name) for each image the model image was compared to.
    """
    modelPrecompValues = open_cached(
        query_fish,
        cache_dir=cache_dir,
    )
    if modelPrecompValues is None:
        if verbose:
            print("Not enough points")
        return []

    centerTranslationMatrix = np.float32([[1, 0, 250], [0, 1, 250], [0, 0, 1]])

    # targetSpots = cv2.imread(modelDict["spots"], 0)
    targetMask = crop_image(cv2.imread(query_fish.mask_path, 0))
    targetMaskShifted = cv2.warpPerspective(
        targetMask,
        centerTranslationMatrix,
        (targetMask.shape[1] + 500, targetMask.shape[0] + 500),
    )

    if verbose:
        print(query_fish.uuid)
        visualize(
            modelPrecompValues["spots_standardised"],
            modelPrecompValues["spots_standardised"],
            annotateOrder=True,
            invertYAxis=False,
            figsize=(10, 10),
        )

    ranking = []
    for fish in fishToComare:
        if progress or verbose:
            print(fish.uuid)

        dataPrecompValues = open_cached(
            fish,
            cache_dir=cache_dir,
        )

        if dataPrecompValues is None:
            ranking.append((-1, 0, fish.uuid))
            continue

        if verbose:
            visualize(
                dataPrecompValues["spots_standardised"],
                dataPrecompValues["spots_standardised"],
                annotateOrder=True,
                invertYAxis=False,
                figsize=(10, 10),
            )

        T, _, score, s_idx, t_idx = aamatch(
            dataPrecompValues["spots_standardised"],
            modelPrecompValues["spots_standardised"],
            dataPrecompValues["invariants"],
            dataPrecompValues["asterisms"],
            dataPrecompValues["kdtree"],
            modelPrecompValues["invariants"],
            modelPrecompValues["asterisms"],
            modelPrecompValues["kdtree"],
        )

        if progress or verbose:
            print(f"Score: {score}")

        maskCoverage = 0
        # spotsCoverage = 0
        if T is not None:
            if verbose:
                transformedCoords = []
                for coord in dataPrecompValues["spots_standardised"]:
                    coordNew = np.array([coord[0], coord[1], 1])
                    coordNew = T.params @ coordNew
                    transformedCoords.append(coordNew[:2])

                # maxValue = max(
                #     np.max(dataPrecompValues["spots_standardised"]),
                #     np.max(dataPrecompValues["spots_standardised"]),
                #     np.max(np.asarray(transformedCoords)),
                # )
                visualize(
                    dataPrecompValues["spots_standardised"],
                    dataPrecompValues["spots_standardised"],
                    annotateOrder=True,
                    invertYAxis=False,
                    figsize=(10, 10),
                )
                visualize(
                    modelPrecompValues["spots_standardised"],
                    modelPrecompValues["spots_standardised"],
                    annotateOrder=True,
                    invertYAxis=False,
                    figsize=(10, 10),
                )

                visualize(
                    modelPrecompValues["spots_standardised"],
                    np.asarray(transformedCoords),
                    annotateOrder=False,
                    invertYAxis=False,
                    figsize=(10, 10),
                )

            model = skimage.transform.AffineTransform()
            model.estimate(
                np.asarray(dataPrecompValues["spots"])[s_idx],
                np.asarray(modelPrecompValues["spots"])[t_idx],
            )
            dataMask = crop_image(cv2.imread(fish.mask_path, 0))

            warped = cv2.warpPerspective(
                dataMask,
                centerTranslationMatrix @ model.params,
                (targetMaskShifted.shape[1], targetMaskShifted.shape[0]),
            )

            pMask, rMask = get_average_precision_recall(
                np.array([warped]), np.array([targetMaskShifted]), verbose=verbose
            )
            maskCoverage = 2 * (pMask * rMask) / (pMask + rMask)

        ranking.append((score * maskCoverage, maskCoverage, fish.uuid))
    return ranking
