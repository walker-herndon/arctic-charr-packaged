##https://bitbucket.org/sergiopr/gmatch/src/default/

import collections
import itertools
import json
import logging
import math
import os
import pickle

import cv2
import numpy as np

# cKDTree does not supoort
# query ball
from scipy.spatial import cKDTree, distance_matrix

from .util import (
    visualize,
    crop_image,
    get_normalise_direction_matrix,
)

_logger = logging.getLogger("gmatch")

__cache_dir__ = "groth_cache"


def gmatch(
    data,
    model,
    reject_scale=10.0,
    theta_max=10 * math.pi / 180,
    c_max=0.99,
    local_triangle_k=0,
    verbose=False,
):
    if isinstance(data, dict):
        cat1 = data["filteredSpots"]
    else:
        cat1 = data
        data = None
    if isinstance(model, dict):
        cat2 = model["filteredSpots"]
    else:
        cat2 = model
        model = None
    if verbose:
        _logger.setLevel(logging.INFO)
    else:
        _logger.setLevel(logging.WARNING)
    step = 0
    maxstep = 1
    ic1 = cat1
    ic2 = cat2

    while True:
        _logger.info("number of points cat1 %i", len(ic1))
        _logger.info("number of points cat2 %i", len(ic2))
        maxmatch = min(ic1.shape[0], ic2.shape[0])
        if maxmatch < 1:
            _logger.warning("no possible matches between catalogues")
            return (None, 0)
        _logger.info("maximum number of matches %i", maxmatch)
        pm, matchScore = gmatch_once(
            ic1,
            ic2,
            reject_scale,
            theta_max,
            c_max,
            local_triangle_k=local_triangle_k,
            first=(step == 0),
            model=model,
            data=data,
        )

        nmatches = len(pm)
        _logger.info("matches: %i, maximum matches: %i", nmatches, maxmatch)
        if len(pm) > 0 and verbose:
            visualize(cat1, ic1[pm[:, 0]])
            visualize(cat2, ic2[pm[:, 1]])
        if nmatches == maxmatch:
            _logger.info(
                "objects in current input catalogues are all matched in %i steps", step
            )
            return (ic1[pm[:, 0]], ic2[pm[:, 1]]), matchScore

        elif nmatches == 0:
            _logger.info("no matches in catalogues")
            return (None, 0)
        elif nmatches < maxmatch:
            if step < maxstep:
                _logger.info("number of matches < number of maximum matches")
                _logger.info("starting over with only matched points")

                ic1 = cat1[pm[:, 0]]
                ic2 = cat2[pm[:, 1]]
                step += 1
            else:
                return (ic1[pm[:, 0]], ic2[pm[:, 1]]), matchScore

        else:
            _logger.error("something is very wrong")
            return (None, 0)


def gmatch_once(
    cat1s,
    cat2s,
    reject_scale=10.0,
    theta_max=10 * math.pi / 180,
    c_max=0.99,
    local_triangle_k=0,
    first=False,
    model=None,
    data=None,
):
    tl1 = None
    tl2 = None
    matches1 = None
    if (not first) or model is None or data is None:
        _logger.info("generating triangles in catalogue 1")
        tl1 = (
            list(create_triang(cat1s, reject_scale=reject_scale, c_max=c_max))
            if local_triangle_k <= 0
            else list(
                create_triang_local(
                    cat1s, reject_scale=reject_scale, c_max=c_max, k=local_triangle_k
                )
            )
        )
        c = cat1s.shape[0]
        _logger.info("expected triangles %i", c * (c - 1) * (c - 2) / 6)
        _logger.info("created triangles %i", len(tl1))

        _logger.info("generating triangles in catalogue 2")
        tl2 = (
            list(create_triang(cat2s, reject_scale=reject_scale, c_max=c_max))
            if local_triangle_k <= 0
            else list(
                create_triang_local(
                    cat2s, reject_scale=reject_scale, c_max=c_max, k=local_triangle_k
                )
            )
        )

        c = cat2s.shape[0]
        _logger.info("expected triangles %i", c * (c - 1) * (c - 2) / 6)
        _logger.info("created triangles %i", len(tl2))
        if len(tl1) == 0 or len(tl2) == 0:
            return ([], 0)

        matches1 = match_quick(
            tl1, tl2, theta_max, reject_scale=reject_scale, model=None, data=None
        )
    else:
        tl1 = data["triangles"]
        tl2 = model["triangles"]
        matches1 = match_quick(
            tl1, tl2, theta_max, reject_scale=reject_scale, model=model, data=data
        )

    _logger.info("we have %i matches", len(matches1))
    if not matches1:
        _logger.info("no matches between the catalogues")
        return ([], 0)

    # removing false matches
    _logger.info("filtering matches")
    matches = clean_matches(matches1)
    # voting to select matched coordinates
    _logger.info("voting")
    pm, matchedScore = votes(matches, cat1s.shape[0], cat2s.shape[0])
    matchedScore = matchedScore * len(matches) / (len(tl1) + len(tl2))
    return (pm, matchedScore)


# Not sure why these are passed but would be a lot to change so going to leave it
# pylint: disable=unused-argument
def match_quick(tl1, tl2, theta_max, reject_scale=10.0, model=None, data=None):
    tspace1 = None
    tspace2 = None
    # kdtree = None
    if model is None or data is None:
        tspace1 = np.array([[tl.R, tl.C, tl.theta] for tl in tl1])
        tspace2 = np.array([[tl.R, tl.C, tl.theta] for tl in tl2])

        kdtree2 = cKDTree(tspace2)
        kdtree1 = cKDTree(tspace1)
        _logger.info("Querying KD-tree")

    else:
        # Need to fix
        tspace1 = data["tspace"]
        tspace2 = model["tspace"]
        kdtree1 = data["kdtree"]
        kdtree2 = model["kdtree"]

    _logger.debug("query in tree...")
    _, r1 = kdtree2.query(tspace1, k=2)
    _, r2 = kdtree1.query(tspace2, k=2)

    _logger.debug("done")

    # dropOffCurve = d1[:, 0] / d1[:, 1]
    # idxs = np.argsort(dropOffCurve)

    matches1 = []
    _logger.info("checking matches")

    for i, r1val in enumerate(r1):
        location = r1val[0]
        # triangle in first catalogue
        t1 = tl1[i]
        if location < len(r2) and i == r2[location][0]:
            t2 = tl2[location]
            matches1.append(
                MatchedTriangles(t1, t2, t1.hel * t2.hel, t1.logp - t2.logp)
            )

    return matches1


_logger = logging.getLogger("gmatch")
TRIANGLE_PARAMS = [
    "v0",
    "v1",
    "v2",
    "i0",
    "i1",
    "i2",
    "logp",
    "hel",
    "R",
    "C",
    "theta",
    "s",
    "centroid",
    "direction_x_1",
    "direction_y_1",
    "direction_x_2",
    "direction_y_2",
    "direction_x_3",
    "direction_y_3",
]
Triangle = collections.namedtuple("Triangle", TRIANGLE_PARAMS)

MatchedTriangles = collections.namedtuple(
    "MatchedTriangles", ["t0", "t1", "hel", "logm"]
)


def norma(x):
    n = np.sqrt(np.dot(x, x.conj()))
    return n


def votes(matches, c1, c2, allow_single_drop=False):
    # shape of the catalogues, not of the matches
    vot = np.zeros((c1, c2), dtype="int")
    # to store matched points
    lm1 = []
    lm2 = []

    for m in matches:
        t0 = m.t0
        t1 = m.t1

        vot[t0.i0, t1.i0] += 1
        vot[t0.i1, t1.i1] += 1
        vot[t0.i2, t1.i2] += 1

    vmax = vot.max()
    _logger.info("maximum voting count %i", vmax)
    if vmax <= 0:
        _logger.info("voting is 0, no match between catalogues")
        return (np.array([]), 0)

    sortv = np.argsort(vot, axis=None)

    # This is beyond my ability to fix
    # pylint: disable=unbalanced-tuple-unpacking
    id0, id1 = np.unravel_index(sortv[::-1], (c1, c2))
    matchedScore = 0

    for i, j in zip(id0, id1):
        val = vot[i, j]
        if val <= 0:
            # votes are 0
            _logger.info("votes have reached 0 level, ending")
            break

        if 2 * val < vmax:
            _logger.info("votes are a half of the maximum, ending")
            # votes are a half of the maximum
            break
        if (i in lm1) or (j in lm2):
            # the point is already matched

            _logger.info("point %i %i already matched, ending (votes %i)", i, j, val)
            break

        _logger.debug("obj %i in cat1 is matched with obj %i in cat2", i, j)
        lm1.append(i)
        lm2.append(j)
        matchedScore += val

    result = np.array([lm1, lm2]).T
    return result, matchedScore


def clean_matches(matches):
    matches = [m for m in matches if m.hel > 0]
    magnitudes = np.array([m.logm for m in matches])
    _logger.info("matches were %i", len(matches))
    magMean = magnitudes.mean()
    magStd = magnitudes.std()
    # magVar = magnitudes.var()
    newMatches = [m for m in matches if abs(m.logm - magMean) <= 1.5 * magStd]
    _logger.info("matches are %i", len(newMatches))
    return newMatches


def create_triang(vlist, reject_scale=10, c_max=0.99):
    for idx in itertools.combinations(range(vlist.shape[0]), 3):
        t = create_triang_(vlist, idx)
        if t.R < reject_scale and t.C < c_max:
            yield t


def create_triang_local(vlist, reject_scale=10, c_max=0.99, k=10):
    kdTree = cKDTree(vlist)
    usedCombinations = set([])
    for i, v in enumerate(vlist):
        _, indexes = kdTree.query(v, k=k)

        indexes = indexes[1:]
        for idx in itertools.combinations(indexes, 2):
            id1, id2 = idx
            idx = tuple(sorted([id1, id2, i]))
            if idx not in usedCombinations and idx[2] < len(vlist):
                usedCombinations.add(idx)
                t = create_triang_(vlist, idx)
                if abs(t.R) < reject_scale and abs(t.C) < c_max:
                    yield t


def create_triang_(vlist, idx):
    v = vlist[idx, :]
    directions = []
    for vertex in v:
        diffVectors = vlist - vertex
        directions.append(np.sum(diffVectors, axis=0) / len(vlist))

    # np.roll(v[:,0:2],shift=-1,axis=0)
    a = v[[1, 2, 0], 0:2] - v[:, 0:2]  # 1-0, 2-1, 0-2
    # norms of the sides
    n = [norma(ar) for ar in a]
    # perimeter
    p = sum(n)
    ll = [(ni, ai, ids, (ids + 1) % 3) for ni, ai, ids in zip(n, a, range(3))]
    ls = sorted(ll, key=lambda x: x[0])
    sides, edges, _, _ = zip(*ls)

    # ov = v[idxs, :]
    oa = np.array(edges)

    # cross product of sides
    e = np.cross(oa, np.roll(oa, shift=-1, axis=0))

    sg = np.sign(e)
    if np.any(sg != sg[0]):
        _logger.info("reorder")
    R = sides[2] / sides[0]
    C = np.dot(oa[0], oa[2]) / (sides[2] * sides[0])
    centroid = (
        (1 / 3) * (v[0][0] + v[1][0] + v[2][0]),
        (1 / 3) * (v[0][1] + v[1][1] + v[2][1]),
    )
    theta = math.atan2((v[0][1] - centroid[1]), (v[0][0] - centroid[0]))
    return Triangle(
        v[0],
        v[1],
        v[2],
        idx[0],
        idx[1],
        idx[2],
        math.log(p),
        sg[0] if sg[0] != 0 else 1,
        R,
        C,
        theta,
        sides[2],
        centroid,
        directions[0][0],
        directions[0][1],
        directions[1][0],
        directions[1][1],
        directions[2][0],
        directions[2][1],
    )


def euclidian_distance(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def standardise_points(points):
    minValues = np.amin(points, axis=0)

    points = np.apply_along_axis(
        lambda p: np.asarray(((p[0] - minValues[0]), (p[1] - minValues[1]))), 1, points
    )

    maxValue = np.amax(points)
    return points / maxValue


def select_points(pointList, tol_e):
    distances = distance_matrix(pointList, pointList)
    commonIndexes = np.argwhere(distances < tol_e * 3)
    indexesToRemove = set()
    for index in commonIndexes:
        if (
            index[0] != index[1]
            and index[0] not in indexesToRemove
            and index[1] not in indexesToRemove
        ):
            indexesToRemove.add(max(index[0], index[1]))
    return np.delete(pointList, list(indexesToRemove), 0)


def findClosestMatch(
    query_fish,
    fishToCompare,
    cache_dir="groth_cache",
    verbose=False,
    progress=False,
    local_triangle_k=0,
):
    """Uses the Groth matching algorithm to find the closest match for the query fish image in the list of fish images to compare.

    Args:
        query_fish (Fish): The query fish image to find the closest match for.
        fishToCompare (List[Fish]): The list of fish images to compare to.
        cache_dir (str, optional): The directory to store cached precomputed values. Defaults to "groth_cache".
        verbose (bool, optional): If True, print verbose information to stdout. Defaults to False.
        progress (bool, optional): If True, print progress information to stdout. Defaults to False.
        local_triangle_k (int, optional): The number of local triangles to use for matching. Defaults to 0.

    Returns:
        Tuple[float, float, str]: The score, mask coverage, and image name of the closest match.
    """
    reject_scale = 10000000
    theta_max = 10 * math.pi / 180
    c_max = 1.01

    precomputedModelValues = None
    if query_fish.precomp is not None:
        with open(query_fish.precomp, "rb") as f:
            precomputedModelValues = pickle.load(f)
    else:
        # Try retrieving using local cache
        precomputedModelValues = retrieve_from_cache(query_fish.uuid, cache_dir)
    if (
        precomputedModelValues is None
        or precomputedModelValues["reject_scale"] != reject_scale
        or precomputedModelValues["c_max"] != c_max
        or "local_triangle_k" not in precomputedModelValues
        or precomputedModelValues["local_triangle_k"] != local_triangle_k
    ):
        precomputedModelValues = precomputeValues(
            query_fish,
            reject_scale,
            c_max,
            cache_dir,
            local_triangle_k,
        )
    elif verbose:
        print("Cache hit")

    if precomputedModelValues is None:
        if progress or verbose:
            print("Not enough points")
        return []

    if verbose:
        visualize(
            precomputedModelValues["standardisedSpots"],
            precomputedModelValues["filteredSpots"],
        )
    ranking = []
    for fish in fishToCompare:
        if progress or verbose:
            print(fish.uuid)

        precomputedFishValues = None
        if fish.precomp is not None:
            with open(fish.precomp, "rb") as f:
                precomputedModelValues = pickle.load(f)
        else:
            # Try retrieving using local cache
            precomputedFishValues = retrieve_from_cache(fish.uuid, cache_dir)
        if (
            precomputedFishValues is None
            or precomputedFishValues["reject_scale"] != reject_scale
            or precomputedFishValues["c_max"] != c_max
            or "local_triangle_k" not in precomputedFishValues
        ):
            precomputedFishValues = precomputeValues(
                fish,
                reject_scale,
                c_max,
                cache_dir,
                local_triangle_k,
            )

        if precomputedFishValues is None:
            if progress or verbose:
                print("Not enough points")
            ranking.append((-1, -1, fish.uuid))
            continue

        if verbose:
            visualize(
                precomputedFishValues["standardisedSpots"],
                precomputedFishValues["filteredSpots"],
            )

        matches, matchScore = gmatch(
            precomputedFishValues,
            precomputedModelValues,
            reject_scale=reject_scale,
            theta_max=theta_max,
            c_max=c_max,
            verbose=verbose,
            local_triangle_k=local_triangle_k,
        )
        if matches:
            if progress or verbose:
                print("Matches", len(matches[0]), matchScore)

        else:
            if progress or verbose:
                print("No match found")

        ranking.append(
            (matchScore, (len(matches[0]) if matches is not None else 0), fish.uuid)
        )
    return ranking


def retrieve_from_cache(imgKey, cache_dir="groth_cache"):
    if os.path.isfile(os.path.join(cache_dir, imgKey + ".pickle")):
        with open(os.path.join(cache_dir, imgKey + ".pickle"), "rb") as f:
            precomputedModelValues = pickle.load(f)
        return precomputedModelValues
    return None


def precomputeValues(
    fish,
    reject_scale,
    c_max,
    cache_dir="groth_cache",
    local_triangle_k=0,
):
    """
    Precomputes values for the given fish.

    Args:
        fish (Fish): The fish object.
        reject_scale (float): The reject scale value.
        c_max (float): The maximum value for c.
        cache_dir (str, optional): The directory to store the precomputed values. Defaults to "groth_cache".
        local_triangle_k (int, optional): The value of k for local triangles. Defaults to 0.

    Returns:
        dict: A dictionary containing the precomputed values.
    """
    spots = None
    if fish.spotJson is not None:
        with open(fish.spotJson, "r", encoding="utf-8") as f:
            spots = json.load(f)
    if spots is not None and len(spots) > 5:
        spots = [n[:2] for n in spots]  # Remove size from spots
        spots = [
            spots[i] for i in range(len(spots)) if i == 0 or spots[i] != spots[i - 1]
        ]  # Remove duplicates

        mask = crop_image(cv2.imread(fish.mask_path, 0))
        R = get_normalise_direction_matrix(mask)
        tmpPoints = np.copy(np.asarray(spots))
        tmpPoints = np.dot(R[:, (0, 1)], np.array([tmpPoints[:, 0], tmpPoints[:, 1]]))
        tmpPoints = np.array(list(zip(tmpPoints[0], tmpPoints[1])))
        warpedTargetMask = crop_image(
            cv2.warpAffine(mask, R, (mask.shape[1] + 500, mask.shape[0] + 500))
        )
        spotsSparse = tmpPoints / np.max(warpedTargetMask.nonzero())

        triangles = (
            list(create_triang(spotsSparse, reject_scale=reject_scale, c_max=c_max))
            if local_triangle_k <= 0
            else list(
                create_triang_local(
                    spotsSparse,
                    reject_scale=reject_scale,
                    c_max=c_max,
                    k=local_triangle_k,
                )
            )
        )

        tspace = np.array([[tl.R, tl.C, tl.theta] for tl in triangles])
        kdtree = cKDTree(tspace)

        precomputedObject = {
            "reject_scale": reject_scale,
            "c_max": c_max,
            "filteredSpots": spotsSparse,
            "triangles": triangles,
            "tspace": tspace,
            "kdtree": kdtree,
            "local_triangle_k": local_triangle_k,
            "version": 1,
        }
        with open(os.path.join(cache_dir, fish.uuid + ".pickle"), "wb") as f:
            pickle.dump(precomputedObject, f)
        return precomputedObject
    else:
        return None
