import json
import sys
import time
from enum import Enum

import numpy as np

from . import DBUtil, astroalignMatch, grothMatcherCustom


class Algorithm(Enum):
    RANSAC_AFFINE = 1
    MODIFIED_GROTH = 2


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


cave = 21
algorithm = Algorithm.RANSAC_AFFINE

resultFileName = f"results.{'aa' if algorithm == Algorithm.RANSAC_AFFINE else 'customGroth'}.C{cave}.json"
print(resultFileName)


totalResults = {}
prevDir = [2012, "June"]
currentDir = [2012, "Aug"]


for i in range(15):
    print(currentDir, prevDir)
    testingImages = DBUtil.get_images(
        cave,
        years=[currentDir[0]],
        months=[currentDir[1]],
        rootDirs=["../all_images/", "patched/results/"],
    )
    databaseImages = DBUtil.get_images(
        cave,
        years=range(2012, prevDir[0] + 1),
        lastMonth=0 if prevDir[1] == "June" else 1,
        rootDirs=["../all_images/", "patched/results/"],
    )
    tagToFish, fishToTag = DBUtil.connectFish(
        cave,
        years=range(2012, currentDir[0] + 1),
        lastMonth=0 if currentDir[1] == "June" else 1,
        rootDirs=["../all_images/", "patched/results/"],
    )
    totalDirectoryProcessingTime = 0
    # pylint: disable=C0206, C0201
    for keyPath in testingImages.keys():
        potentialMatches = tagToFish[fishToTag[keyPath]]
        potentialMatches.remove(keyPath)
        print(
            "Testing: "
            + str(keyPath)
            + " ("
            + str(fishToTag[keyPath])
            + "), Potential matches in DB: "
            + str(potentialMatches)
        )
        sys.stdout.flush()
        currentTime = time.time()
        results = None
        if algorithm == Algorithm.RANSAC_AFFINE:
            results = astroalignMatch.findClosestMatch(
                keyPath,
                testingImages[keyPath],
                databaseImages,
                verbose=False,
                progress=False,
            )
        elif algorithm == Algorithm.MODIFIED_GROTH:
            results = grothMatcherCustom.findClosestMatch(
                keyPath,
                testingImages[keyPath],
                databaseImages,
                progress=False,
                local_triangle_k=25,
            )  # Added local_triangle_k
        else:
            print("Unrecognised algorithm")

        elapsedTime = time.time() - currentTime
        totalDirectoryProcessingTime += elapsedTime
        results = [list(entry) for entry in results]
        # everything commented out is just used to get "order" value (not sure what it is) and print it
        # [entry.append(fishToTag[entry[-1]]) for entry in results]

        #         order = sorted(results, key=lambda x: x[0], reverse = True)

        #         order = [(entry[0], entry[1], fishToTag[entry[1]]) for entry in order if entry[0] != 0]
        #         #order = [(entry[0], entry[1], fishToTag[entry[1]], entry[2]) for entry in order if entry[0] != 0]
        print(f"{round(elapsedTime, 2)} seconds to process matches")
        #         print(order)
        sys.stdout.flush()
        totalResults[keyPath] = {
            "tag": fishToTag[keyPath],
            "potentialMatches": potentialMatches,
            "results": results,
        }

        with open(resultFileName, "w", encoding="utf-8") as f:
            json.dump(totalResults, f, cls=NpEncoder)

    averageDirectoryTime = totalDirectoryProcessingTime / len(testingImages)
    print(
        "Directory finished: ",
        currentDir[0],
        "/",
        currentDir[1],
        " ",
        totalDirectoryProcessingTime,
        averageDirectoryTime,
        len(databaseImages),
    )
    sys.stdout.flush()

    prevDir[0] = currentDir[0]
    prevDir[1] = currentDir[1]

    currentDir[0] = currentDir[0] + 1 if currentDir[1] == "Aug" else currentDir[0]
    currentDir[1] = "June" if currentDir[1] == "Aug" else "Aug"

print("Finished")
