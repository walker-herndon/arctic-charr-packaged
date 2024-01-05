import os

from . import UnetExtractor, astroalignMatch, grothMatcherCustom
from .algorithms import Algorithm
from .fish import Fish


def _pathToKey(path: str):
    """Default function for translating an image path to a key. Assumes path is of form
    <image root>/<year>_<month>/Cave<#>/<image name>.JPG

    path (str|Path) The path to translate

    Returns: (str) key of form `C<#>-<year>-<month>-<image name>`
    """
    # Split into cave, year, month, img name
    pathElements = path.split(os.sep)
    cave = pathElements[-2][4:]
    year = pathElements[-3].split("_")[0]
    month = pathElements[-3].split("_")[1]
    imgName = pathElements[-1].split(".")[0]
    return f"C{cave}-{year}-{month}-{imgName}"


def _translatePath(fish):
    """Default function for translating an original image path to a result path for spots and masks. Creates intermediate folders if no such end folder exists.
    inPath (str|Path) The path to translate

    Returns: (str) The translated path
    """
    outPath = f"results{os.sep}{os.sep.join(fish.image_path.split(os.sep)[-3:])}"
    if not os.path.exists(os.path.dirname(outPath)):
        os.makedirs(os.path.dirname(outPath))
    return outPath


def translatePathUnsorted(fish):
    """Alternative default function for translating an original image path to a result path for spots and masks. Uses flat folder structure with UUIDs as file names.
    inPath (str|Path) The path to translate

    Returns: (str) The translated path
    """
    outPath = f"results{os.sep}{fish.uuid}"
    if not os.path.exists(os.path.dirname(outPath)):
        os.makedirs(os.path.dirname(outPath))
    return outPath


def _imgDirectoryTranslatorWrapper(root):
    def _imgDirectoryTranslator(key):
        """Default function for translating a key to an image directory
        key (str|Path) The key to convert to a file path

        Returns: (str|None) the filepath assosciatred with the key
        """
        # Split into cave, year, month, img name
        keyElements = key.split("-")
        dirPath = os.path.join(
            root,
            f"{keyElements[1]}_{keyElements[2]}",
            f"Cave{keyElements[0][1:]}",
        )
        # Can have extension of .jpg, .JPG, .JPEG, so just check which one it is. Ignore hidden files (starting with .)
        filteredImageNames = [
            path
            for path in os.listdir(dirPath)
            if keyElements[3] in path and "." != path[0]
        ]
        if len(filteredImageNames) > 0:
            return os.path.join(dirPath, filteredImageNames[0])
        return None

    return _imgDirectoryTranslator


class Matcher:
    """
    Matcher class for matching images
    """

    def __init__(
        self,
        imgRoot="all_images",
        grothCache="groth_cache",
        astroalignCache="aa_cache",
        maskModelWeights="unet_mask.hdf5",
        maskBatchSize=20,
        maskResultOutputDir=None,
        maskFileSuffix=".mask.png",
        customMaskExtractor=None,
        spotUnetWeights="unet_spots.hdf5",
        spotBatchSize=10,
        spotThreshold=0.01,
        spotResultOutputDir=None,
        spotFileSuffix=".spots.png",
        spotJsonFileSuffix=".spots.json",
        customSpotExtractor=None,
        gpuAcceleration=True,
        keyToPathTranslator=None,
    ):
        """Constructor for a new matcher
        imgRoot             (str|Path) The directory where the base images are stored. If custom keyToPathTranslator is provided, this has no effect
        grothCache          (str|Path) Directory where the custom groth algorithm should store it's cache files
        astroalignCache     (str|Path) Directory where the astroalign algorithm should store it's cache files
        maskModelWeights    (str|Path) Path to the stored weights for the mask Unet model. Can be omitted if an initialised extractor is provided in the
                                       customMaskExtractor argument.
        maskBatchSize       (int)      The batch size to use when processing masks. Speeds up extraction of multiple masks at once, but uses more memory
                                       (GPU or RAM depending if GPU acceleration is used)
        maskResultOutputDir (func)     Function of the from (str | Path) -> (str | Path) which converts an input image path into a path where the result of the mask
                                       extraction operation should be stored. The default function will mirror the path of the image file, but translatePathUnsorted
                                       can be used instead to produce a flat directory using the fish uuids.
        maskFileSuffix      (str)      The suffix that is used to identify mask files
        customMaskExtractor (Object)   An object that contains a function of form ([str], batch_size=int, outputDir=func (str | Path) -> (str | Path)) -> None
                                       which extracts a mask from the original image to the location indicated by the function passed as the outputDir with the suffix
                                       given in maskFileSuffix argument
        spotUnetWeights     (str|Path) Path to the stored weights for the spot Unet model. Can be omitted if an initialised extractor is provided in the
                                       customSpotExtractor argument.
        spotBatchSize       (int)      The batch size to use when processing spots. Speeds up extraction of spots from multiple fish at once, but uses more memory
                                       (GPU or RAM depending if GPU acceleration is used)
        spotThreshold       (float)    Value in range [0,1] that is applied to the spots prediction to as a threshold for determining which part of the prediction should
                                       be taken as spots. Higher values result in higher precision, lower values result in higher recall.
        spotResultOutputDir (func)     Function of the from (str | Path) -> (str | Path) which converts an input image path into a path where the result of the spot
                                       extraction operation should be stored
        spotFileSuffix      (str)      The suffix that is used to identify raw spot files
        spotJsonFileSuffix  (str)      The suffix that is used to identify json spot files
        customSpotExtractor (Object)   An object that contains a function of form
                                       ([str], [str], batch_size=int, threshold=float, outputDir=func (str | Path) -> (str | Path)) -> None
                                       which extracts spots from the original image and mask to the location indicated by the function passed as the outputDir with the
                                       suffix given in spotFileSuffix and spotJsonFileSuffix arguments
        gpuAcceleration     (bool)     If to use CUDA acceleration when extracting masks an spots. If true, uses the first available GPU device, else it uses the
                                       first available CPU device. Can be ommitted if passing a custom spot and mask extractor.
        keyToPathTranslator (func)     Function of form (str) -> (str | Path) which takes in an image key, and converts it to the path of the location of the base image.
        """

        # check that cache folders exist and create them if not
        if not os.path.exists(grothCache):
            os.makedirs(grothCache)
        if not os.path.exists(astroalignCache):
            os.makedirs(astroalignCache)

        self.grothCache = grothCache
        self.astroalignCache = astroalignCache
        self.keyToPathTranslator = (
            _imgDirectoryTranslatorWrapper(imgRoot)
            if keyToPathTranslator is None
            else keyToPathTranslator
        )

        self.maskBatchSize = maskBatchSize
        self.maskResultOutputDir = (
            _translatePath if maskResultOutputDir is None else maskResultOutputDir
        )
        self.maskFileSuffix = maskFileSuffix

        self.spotBatchSize = spotBatchSize
        self.spotThreshold = spotThreshold
        self.spotResultOutputDir = (
            _translatePath if spotResultOutputDir is None else spotResultOutputDir
        )
        self.spotFileSuffix = spotFileSuffix
        self.spotJsonFileSuffix = spotJsonFileSuffix

        # If a custom spot or mask extractor was passed, use that over the default unet extractor. Can be used to supply a Unet extractor with options other than default.
        if customMaskExtractor is not None:
            self.maskExtractor = customMaskExtractor
        else:
            self.maskExtractor = UnetExtractor.UnetMaskExtractor(
                modelWeights=maskModelWeights, useGPU=gpuAcceleration
            )

        if customSpotExtractor is not None:
            self.spotExtractor = customSpotExtractor
        else:
            self.spotExtractor = UnetExtractor.UnetSpotExtractor(
                modelWeights=spotUnetWeights, useGPU=gpuAcceleration
            )

    def matching(
        self,
        query_imgs,
        matching_imgs,
        algorithm=Algorithm.CUSTOM_GROTH,
        rankingLimit=None,
        verbose=False,
    ):
        """Matches a set of Fish objects against another set of Fish objects.

        Args:
            query_imgs    ([Fish]|Fish)          The Fish objects (or Fish object) that are to be matched
            matching_imgs ([Fish]|Fish)          The Fish objects (or Fish object) that are to be matched to
            algorithm     (algorithms.Algorithm) The matching algorithm to use
            rankingLimit  (int|None)             The max number of results to return
            verbose       (bool)                 If True, print progress

        Returns a dictionary of the form:
        {str: [
            {"file_name": str,
             "ranking": int,
             "score": float}
        ]}
        """
        # Support both singular input and multiple input
        if isinstance(query_imgs, Fish):
            query_imgs = [query_imgs]
        if isinstance(matching_imgs, Fish):
            matching_imgs = [matching_imgs]

        all_fish = query_imgs + matching_imgs

        # Check if any images need to have masks extracted
        self.__ensureMasksExtracted(all_fish, verbose)
        # Check if any images need to have spots extracted
        self.__ensureSpotsExtracted(all_fish, verbose)

        results = {}
        # Run the correct algorithm
        if algorithm == Algorithm.CUSTOM_GROTH:
            # grothMatcherCustom.set_cache_dir(self.grothCache)
            for fish in query_imgs:
                if verbose:
                    print(f"Matching {fish.uuid}")
                results[fish.uuid] = []
                result = grothMatcherCustom.findClosestMatch(
                    fish,
                    matching_imgs,
                    cache_dir=self.grothCache,
                    local_triangle_k=25,
                    progress=verbose,
                )

                # Order results
                orderedResult = sorted(result, key=lambda x: x[0], reverse=True)

                # Restrict to number results requested
                limit = rankingLimit
                if limit is None:
                    limit = len(orderedResult)
                for rank in range(min(limit, len(orderedResult))):
                    r = orderedResult[rank]
                    results[fish.uuid].append(
                        {"file_name": r[2], "ranking": rank + 1, "score": r[0]}
                    )
        else:
            # astroalignMatch.set_cache_dir(self.grothCache)
            for fish in query_imgs:
                if verbose:
                    print(f"Matching {fish.uuid}")
                results[fish.uuid] = []
                result = astroalignMatch.findClosestMatch(
                    fish,
                    matching_imgs,
                    cache_dir=self.astroalignCache,
                    progress=verbose,
                )
                # Order results
                orderedResult = sorted(result, key=lambda x: x[0], reverse=True)
                # Restrict to number results requested
                limit = rankingLimit
                if limit is None:
                    limit = len(orderedResult)
                for rank in range(min(limit, len(orderedResult))):
                    r = orderedResult[rank]
                    results[fish.uuid].append(
                        {"file_name": r[2], "ranking": rank + 1, "score": r[0]}
                    )

        if verbose:
            print("Matching complete")
        return results

    def __ensureMasksExtracted(self, fish_list, verbose):
        """Ensures all Fish have extracted masks, extracting the masks where necessary.
        Mask paths can then be accessed by calling fish.mask_path.

        Args:
            fish_list ([Fish]): List of Fish for which to make sure masks exist, extracting masks when no mask is present.
            verbose (bool): If to print verbose information. Defaults to False.

        Returns:
            None
        """

        fishToProcess = []
        for fish in fish_list:
            if fish.image_path is None:
                raise IOError(f"No image found for fish \n{fish}")
            maskResultsPath = self.maskResultOutputDir(fish)
            if fish.mask_path is None:
                fish.mask_path = maskResultsPath + self.maskFileSuffix

            if not os.path.isfile(
                maskResultsPath + self.maskFileSuffix
            ):  # Check if file generated exists
                fishToProcess.append(fish)

        if len(fishToProcess) > 0:
            if verbose:
                print("Extracting masks")
            self.maskExtractor.generate_masks(
                fishToProcess,
                batch_size=self.maskBatchSize,
                verbose=verbose,
            )

    def __ensureSpotsExtracted(self, fish_list, verbose):
        """Ensures all Fish have extracted spots, extracting the spots where necessary.
        Spot paths and spot json paths can then be accessed by calling fish.spot_path and fish.spotJson.

        Args:
            fish_list ([Fish]) List of Fish for which to make sure spots exist, extracting spots when no spot file is present.
            verbose (bool): If to print verbose information. Defaults to False.

        Return:
            None
        """
        fishToProcess = []

        for fish in fish_list:
            if fish.image_path is None:
                raise IOError(f"No image found for fish {fish}")

            spotResultsPath = self.spotResultOutputDir(fish)
            if fish.spot_path is None:
                fish.spot_path = spotResultsPath + self.spotFileSuffix
                fish.spotJson = spotResultsPath + self.spotJsonFileSuffix

            maskResultsPath = self.maskResultOutputDir(fish)

            if not os.path.isfile(
                maskResultsPath + self.maskFileSuffix
            ):  # Check if mask file exists
                raise IOError(f"No mask found when generating spots for fish {fish}")

            if not os.path.isfile(
                spotResultsPath + self.spotJsonFileSuffix
            ) or not os.path.isfile(
                spotResultsPath + self.spotFileSuffix
            ):  # Check if spots file exists
                fishToProcess.append(fish)

        if len(fishToProcess) > 0:
            if verbose:
                print("Extracting spots")
            self.spotExtractor.generate_spots_patched(
                fishToProcess,
                batch_size=self.spotBatchSize,
                threshold=self.spotThreshold,
                verbose=verbose,
            )
