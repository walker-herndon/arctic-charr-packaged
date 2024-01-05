import os
from enum import Enum

import pandas as pd
from .fish import Fish


class DateOrder(Enum):
    YEAR_MONTH = 1
    MONTH_YEAR = 2


def generate_dates(
    years, months, firstMonth=0, lastMonth=1, order=DateOrder.YEAR_MONTH, sep="_"
):
    """Generate dates for the given years and months

    Args:
        years (Iterable[int]): The years for which to generate dates
        months (Iterable[str]): The months for which to generate dates
        firstMonth (int, optional): The index of the month (in the months list) from which to start generating dates in the first year. Defaults to 0.
        lastMonth (int, optional): The index of the month (in the months list) to which the dates shuold be generated in the final year. Defaults to 1.
        order (DateOrder, optional): The order to yield dates. Defaults to DateOrder.YEAR_MONTH.
        sep (str, optional): The seperator to use between components. Defaults to "_".

    Yields:
        str: The specified dates
    """
    for year in years:
        monthIdx = 0
        for month in months:
            if year == years[0] and monthIdx < firstMonth:
                pass
            yield str(year) + sep + str(
                month
            ) if order == DateOrder.YEAR_MONTH else str(month) + sep + str(
                year
            ), year, monthIdx

            if year == years[-1] and monthIdx == lastMonth:
                break
            monthIdx += 1


def _dir_generator(
    caveNum,
    rootDirs=None,
    years=range(2012, 2020),
    months=None,
    firstMonth=1,
    lastMonth=1,
    verbose=False,
):
    """Generates directory pths for the given parameters

    Args:
        caveNum (int): The number of the cave to generate directories for.
        rootDirs (list, optional): The root directories to generate directories from. Defaults to ["../all_images/", "results"].
        years ([type], optional): The years for which to generate directory paths. Defaults to range(2012, 2020).
        months (list, optional): The months for which to generate directoy paths. Defaults to ["June", "Aug"].
        firstMonth (int, optional): The index of the month (in the months list) from which to start generating dates in the first year. Defaults to 1.
        lastMonth (int, optional): The index of the month (in the months list) to which the dates shuold be generated in the final year. Defaults to 1.
        verbose (bool, optional): If to print verbose information. Defaults to False.

    Yields:
        str: Directory paths for the specified dates
    """
    if rootDirs is None:
        rootDirs = ["../all_images/", "results"]
    if months is None:
        months = ["June", "Aug"]
    for rootDir in rootDirs:
        for date, year, monthIdx in generate_dates(
            years=years, months=months, firstMonth=firstMonth, lastMonth=lastMonth
        ):
            directory = os.path.join(rootDir, date, "Cave" + str(caveNum))
            if os.path.isdir(directory):
                yield directory, str(caveNum), year, monthIdx
            elif verbose:
                print(f"{directory} is not a directory")


def _assignToFish(images, uuid, fileName, attribute, value):
    """Helper function to assign value to fish which may not exist"""
    if uuid not in images:
        images[uuid] = Fish(fileName, uuid=uuid)
    setattr(images[uuid], attribute, value)


def generateUUID(rootDirs, path):
    """Transforms a path into a uuid where the root is removed and the path is split on os.sep and joined with a -"""
    root = ""
    for rootDir in rootDirs:
        if path.startswith(rootDir):
            root = rootDir
            break
    pathComponents = path.replace(root, "").split(os.sep)
    pathComponents[-1] = pathComponents[-1].split(".")[0]

    uuid = "-".join(pathComponents)
    # remove any trailing or leading -
    if uuid.startswith("-"):
        uuid = uuid[1:]
    if uuid.endswith("-"):
        uuid = uuid[:-1]
    return uuid


def get_fish(
    caveNum,
    rootDirs=None,
    years=range(2012, 2020),
    months=None,
    firstMonth=0,
    lastMonth=1,
    verbose=False,
):
    """Generate a list of Fish objects for the given parameters

    Args:
        caveNum (int): The number of the cave to generate directories for.
        rootDirs (list, optional): The root directories to generate directories from. Defaults to ["../all_images/", "results"].
        years (range, optional): The years for which to generate directory paths. Defaults to range(2012, 2020).
        months (list, optional): The months for which to generate directoy paths. Defaults to ["June", "Aug"].
        firstMonth (int, optional): The index of the month (in the months list) from which to start generating dates in the first year. Defaults to 0.
        lastMonth (int, optional): The index of the month (in the months list) to which the dates shuold be generated in the final year. Defaults to 1.
        verbose (bool, optional): If to print verbose information. Defaults to False.

    Returns:
        list[Fish]: A list of Fish objects
    """
    if rootDirs is None:
        rootDirs = ["../all_images/", "results"]
    if months is None:
        months = ["June", "Aug"]
    images = {}
    for directory, _, _, _ in _dir_generator(
        caveNum,
        rootDirs=rootDirs,
        years=years,
        months=months,
        firstMonth=firstMonth,
        lastMonth=lastMonth,
        verbose=verbose,
    ):
        for file in sorted(os.listdir(directory)):
            filePath = os.path.join(directory, file)
            if (
                os.path.isfile(filePath)
                and ("IMG" in file or "DSC" in file)
                and (".xcf" not in file)
            ):
                fileComponents = file.split(".")
                fileName = fileComponents[0]
                uuid = generateUUID(rootDirs, filePath)

                if len(fileComponents) == 2 and fileComponents[-1].lower() in [
                    "jpg",
                    "jpeg",
                    "png",
                    "bmp",
                ]:
                    _assignToFish(images, uuid, fileName, "image_path", filePath)
                elif (
                    len(fileComponents) >= 2 and fileComponents[-1].lower() == "pickle"
                ):
                    if len(fileComponents) >= 3 and fileComponents[1] == "aa":
                        _assignToFish(images, uuid, fileName, "precompAA", filePath)
                    else:
                        _assignToFish(images, uuid, fileName, "precomp", filePath)
                elif len(fileComponents) >= 3 and fileComponents[2] == "mask":
                    if len(fileComponents) >= 4 and fileComponents[3] == "acc":
                        _assignToFish(images, uuid, fileName, "maskLabel", filePath)
                    else:
                        _assignToFish(images, uuid, fileName, "mask_path", filePath)
                elif (
                    len(fileComponents) >= 3
                    and fileComponents[2] == "spots"
                    and fileComponents[-1] == "json"
                ):
                    _assignToFish(images, uuid, fileName, "spotJson", filePath)
                elif (
                    len(fileComponents) >= 3
                    and fileComponents[2] == "spots"
                    or fileComponents[1] == "spots"
                ):
                    if (
                        len(fileComponents) >= 4 and fileComponents[3] == "acc"
                    ) or fileComponents[2] == "acc":
                        _assignToFish(images, uuid, fileName, "spotsLabel", filePath)
                    else:
                        _assignToFish(images, uuid, fileName, "spot_path", filePath)

    return list(images.values())


def get_unsorted_fish(rootDirs=None, excludeDirs=None, verbose=False):
    """Generate a list of Fish objects for the given parameters

    Args:
        rootDirs (list, optional): The root directories to generate Fish from. Defaults to ["../all_images/", "results"].
        excludeDirs (list, optional): The subdirectories to exclude. Defaults to None.
        verbose (bool, optional): If to print verbose information. Defaults to False.

    Returns:
        list[Fish]: A list of Fish objects
    """
    if rootDirs is None:
        rootDirs = ["../all_images/", "results"]
    if excludeDirs is None:
        excludeDirs = []
    images = {}
    for rootDir in rootDirs:
        for directory, _, _ in os.walk(rootDir):
            if any(excludeDir in directory for excludeDir in excludeDirs):
                continue
            files = os.listdir(directory)
            if verbose and len(files) > 0:
                print(f"Processing {len(files)} files in {directory}")
            for file in sorted(os.listdir(directory)):
                filePath = os.path.join(directory, file)
                if (
                    os.path.isfile(filePath)
                    and ("IMG" in file or "DSC" in file)
                    and (".xcf" not in file)
                ):
                    fileComponents = file.split(".")
                    fileName = fileComponents[0]
                    uuid = generateUUID(rootDirs, filePath)

                    if len(fileComponents) == 2 and fileComponents[-1].lower() in [
                        "jpg",
                        "jpeg",
                        "png",
                        "bmp",
                    ]:
                        _assignToFish(images, uuid, fileName, "image_path", filePath)
                    elif (
                        len(fileComponents) >= 2
                        and fileComponents[-1].lower() == "pickle"
                    ):
                        if len(fileComponents) >= 3 and fileComponents[1] == "aa":
                            _assignToFish(images, uuid, fileName, "precompAA", filePath)
                        else:
                            _assignToFish(images, uuid, fileName, "precomp", filePath)
                    elif len(fileComponents) >= 3 and fileComponents[2] == "mask":
                        if len(fileComponents) >= 4 and fileComponents[3] == "acc":
                            _assignToFish(images, uuid, fileName, "maskLabel", filePath)
                        else:
                            _assignToFish(images, uuid, fileName, "mask_path", filePath)
                    elif (
                        len(fileComponents) >= 3
                        and fileComponents[2] == "spots"
                        and fileComponents[-1] == "json"
                    ):
                        _assignToFish(images, uuid, fileName, "spotJson", filePath)
                    elif (
                        len(fileComponents) >= 3
                        and fileComponents[2] == "spots"
                        or fileComponents[1] == "spots"
                    ):
                        if (
                            len(fileComponents) >= 4 and fileComponents[3] == "acc"
                        ) or fileComponents[2] == "acc":
                            _assignToFish(
                                images, uuid, fileName, "spotsLabel", filePath
                            )
                        else:
                            _assignToFish(images, uuid, fileName, "spot_path", filePath)

    return list(images.values())


def get_fish_from_paths(paths, rootDirs=None, verbose=False):
    """Generate a list of Fish objects for the given parameters

    Args:
        paths (list[str]): The paths to generate Fish objects from.
        rootDirs (list, optional): The root directories to generate Fish from. Defaults to ["../all_images/", "results"].
        verbose (bool, optional): If to print verbose information. Defaults to False.

    Returns:
        list[Fish]: A list of Fish objects
    """
    if rootDirs is None:
        rootDirs = ["../all_images/", "results"]
    if isinstance(paths, str):
        paths = [paths]
    images = {}
    added_paths = []
    for path in paths:
        # replace rootDir of path with other rootDirs and add to added_paths
        for rootDir in rootDirs:
            if rootDir in path:
                for otherRootDir in rootDirs:
                    if rootDir != otherRootDir:
                        new_path = path.replace(rootDir, otherRootDir)
                        added_paths.append(new_path)

    paths = paths + added_paths

    for path in paths:
        if verbose:
            print(f"Processing {path}")
        fileComponents = path.split(".")
        fileName = fileComponents[0]
        uuid = generateUUID(rootDirs, path)

        if len(fileComponents) == 2 and fileComponents[-1].lower() in [
            "jpg",
            "jpeg",
            "png",
            "bmp",
        ]:
            _assignToFish(images, uuid, path, "image_path", path)
        elif len(fileComponents) >= 2 and fileComponents[-1].lower() == "pickle":
            if len(fileComponents) >= 3 and fileComponents[1] == "aa":
                _assignToFish(images, uuid, fileName, "precompAA", path)
            else:
                _assignToFish(images, uuid, fileName, "precomp", path)
        elif len(fileComponents) >= 3 and fileComponents[2] == "mask":
            if len(fileComponents) >= 4 and fileComponents[3] == "acc":
                _assignToFish(images, uuid, fileName, "maskLabel", path)
            else:
                _assignToFish(images, uuid, fileName, "mask_path", path)
        elif (
            len(fileComponents) >= 3
            and fileComponents[2] == "spots"
            and fileComponents[-1] == "json"
        ):
            _assignToFish(images, uuid, fileName, "spotJson", path)
        elif (
            len(fileComponents) >= 3
            and fileComponents[2] == "spots"
            or fileComponents[1] == "spots"
        ):
            if (
                len(fileComponents) >= 4 and fileComponents[3] == "acc"
            ) or fileComponents[2] == "acc":
                _assignToFish(images, uuid, fileName, "spotsLabel", path)
            else:
                _assignToFish(images, uuid, fileName, "spot_path", path)

    return list(images.values())


def get_fish_from_uuid(uuid, fish_list):
    for fish in fish_list:
        if fish.uuid == uuid:
            return fish
    return None


def connectFish(
    caveNum,
    rootDirs=None,
    years=range(2012, 2020),
    months=None,
    tagTranslation=None,
    firstMonth=0,
    lastMonth=1,
    verbose=False,
):
    """Connects fish based on csv files in the given directories"""
    if rootDirs is None:
        rootDirs = ["../all_images/", "results"]
    if months is None:
        months = ["June", "Aug"]
    if tagTranslation is None:
        tagTranslation = {}
    fish = {}
    inverseMapping = {}
    for directory, cave, year, monthIdx in _dir_generator(
        caveNum,
        rootDirs=rootDirs,
        years=years,
        months=months,
        firstMonth=firstMonth,
        lastMonth=lastMonth,
        verbose=verbose,
    ):
        metadataFound = False
        for file in os.listdir(directory):
            filePath = os.path.join(directory, file)
            if os.path.isfile(filePath) and (
                file.startswith("filename") or file.startswith("newfilename")
            ):
                df = None
                if file.split(".")[-1].lower() == "csv":
                    df = pd.read_csv(filePath, delimiter=";", quotechar="'")
                elif file.split(".")[-1].lower() == "xlsx":
                    df = pd.read_excel(filePath, sheet_name="Sheet1")

                if df is not None:
                    if verbose:
                        print(filePath)
                    try:
                        for index, row in df.iterrows():
                            tag = str(row["ID"])
                            if tag.lower() == "nan":
                                if (
                                    "comments" in row
                                    and str(row["comments"]).lower() != "nan"
                                ):
                                    tag = f"c:{str(row['comments'])}"
                                else:
                                    tag = None
                            if tagTranslation is not None and tag in tagTranslation:
                                tag = tagTranslation[tag]

                            if str(row["filename"]).lower() == "nan":
                                if verbose:
                                    print(
                                        f"Row {index} in metadatafile {filePath} does not list an image filename!"
                                    )
                            else:
                                keyPath = f"C{cave}-{str(year)}-{months[monthIdx]}-{row['filename'].split('.')[0]}"
                                ## !!------------------ File entry in CSV file is wrong. Fix here as cannot modfiy source files --------------------!! ##
                                if keyPath == "C21-2017-Aug-IMG_3262":
                                    keyPath = "C21-2017-Aug-IMG_3263"
                                ## !!------------------                            End of fix                                   --------------------!! ##
                                inverseMapping[keyPath] = tag
                                try:
                                    fish[tag].append(keyPath)
                                except KeyError:
                                    fish[tag] = [keyPath]
                        metadataFound = True
                    # didn't write the code so not sure what exceptions this expects to be thrown
                    # pylint: disable=broad-except
                    except Exception:
                        if verbose:
                            print(
                                "Error occured. Looking for different metadata file..."
                            )
                        metadataFound = False
                if metadataFound:
                    break

        if not metadataFound and verbose:
            print(f"Cannot find metadata file for directory: {directory}")

    return fish, inverseMapping
