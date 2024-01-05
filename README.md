## Re-identification of individuals from images using spot constellations; a case study in Arctic charr (Salvelinus alpinus)

Ignacy T. Dębicki, Elizabeth A. Mittell, Bjarni K. Kristjánsson, Camille A. Leblanc, Michael B. Morrissey, & Kasim Terzić

Modified by Walker Herndon to transform the code into a python library

#### Instructions for installation

You will need to have python and pip installed to use the library. You can check that these are installed by running `python --version` and `pip --version` in the terminal. 

Depending on how these were installed on your computer you might have to run `python3 --version` and `pip3 --version` instead.

If this command lists version number you are good to go, otherwise install Python and Pip now.

##### Mac and Linux

With both Python and Pip installed, download the installation script `install.sh`. Once it is downloaded, run it in the terminal using `source ./install.sh path/to/your/project`. This will take a minute to finish running.

Then to use it in a Jupyter Notebook, simply run `jupyter notebook`, go to localhost:8888/tree in your browser, and create a new notebook file. You will need to select `.venv` as your kernel. You can then import and use the library as shown below.

##### Windows

With both Python and Pip installed, download the installation script `install.ps1`. Once it is downloaded, run it in the terminal using `.\setup_venv.ps1 C:\path\to\your\project`. This will take a minute to finish running.

Next run the following:
1. `cd C:path\to\your\project`
2. `.venv\Scripts\Activate.ps1`
3. `jupyter notebook`

Then to use it in a Jupyter Notebook, simply run `jupyter notebook`, go to localhost:8888/tree in your browser, and create a new notebook file. You will need to select `.venv` as your kernel. You can then import and use the library as shown below.

##### Working with virual environments

Using virtual environments when working with Python is good practice as it keeps your dependencies separated between different programs. The installation scripts above will automatically create a virtual environment with everything needed for the matcher installed within it. The virtual environment will be created at the path you provide.

Comprehensive documentation of Python virtual environments, or “venv” can be found [here](https://docs.python.org/3/library/venv.html), but the basics are also provided below. You don't need to worry about this when first installing the matcher as the process is done for you, but when you continue working on the project in the future this information will be useful.

1. To create a new virtual environment run `python -m venv /path/to/venv` where the path can be anywhere you’d like to store the virtual environment. Virtual environment directories are often named something like `.venv` (the dot at the start makes the folder hidden by default), but you can name it whatever you’d like.
2. To use the venv, first activate it by running `source <venv>/bin/activate` where `<venv>` is the path to the venv given in step 1. This command assumes the shell you are using is bash or ash (bash is the default on Mac and Linux so if that doesn’t mean anything to you, you are almost definitely using bash). If you aren’t using bash or zsh the equivalent command for your shell can be found in the documentation linked above.
3. Now that the venv is activated, when you install a library using `pip install <library>` it will be installed within the venv instead of globally. This is why virtual environments are good to keep your dependencies organized.
4. To deactivate the virtual environment again, just run `deactivate`.

If you have previously run the installation script and want to use the matcher again, just go to the terminal, run `cd path/to/your/project` and activate the environment as shown above. Then, just run `jupyter notebook`.

#### Usage

Start by importing some classes from the library
```python
from arctic_charr_matcher.Matcher import Matcher
from arctic_charr_matcher import DBUtil
from arctic_charr_matcher.algorithms import Algorithm
```

Next, initialize the matcher
```python
matcher = Matcher(
	imgRoot="path/to/images",
	maskModelWeights="path/to/mask_model.hdf5",
	spotUnetWeights="path/to/spot_model.hdf5"
)
```
**INCLUDE SOMETHING ABOUT THOSE OTHER PARAMETERS TOO**

You'll need to supply the images to work with and the trained models.

##### Creating Image List

`DBUtil` is used to generate the list of images to be used by the matcher. There are a few options for how to generate the image list which are outlined below. All of these functions return a list of `Fish` objects, each of which are associated with one of the fish images. The list of `Fish` objects is what the matcher takes in.

###### From sorted images

If your images are organized you can provide a date range for the images that should be passed to the matcher. Here's what the file structure for the images needs to look like:

```
└── images/
	├── 2012_June/
	│ ├── Cave1/
	│ │ ├── IMG_001.JPG
	│ │ ├── IMG_002.JPG
	│ │ └── ...
	│ ├── Cave2/
	│ ├── Cave3/
	│ └── ...
	├── 2012_Aug/
	├── 2013_June/
	├── 2013_Aug/
	└── ...
```

Here is an example of using `DBUtil.get_fish` to get the list of images:

```python
matchingImages = DBUtil.get_fish(21, rootDirs=["path/to/images", "path/to/results"], years=range(2015,2018))
```
Here we are loading in the images to match our query image against. There are a few parameters to consider:
- **Cave number:** I've picked 21 but it can be any cave you want to run the matcher on.
- **rootDirs:** should be the path to the base image directory, and the path to a results directory which **you will need to create in your project folder if it isn't already there**. This directory will hold the image mask and spot files which is what the matcher performs matching on. This should be in the same directory as you are running your script from.
- **years:** the range of years you want to include in `matchingImages` (first value is inclusive, second is exclusive, so this example is `[2015, 2016, 2017]`)

Other parameters:
- **months:** defaults to `["June", "Aug"]` but if your months are different you can provide your own list of months here.
- **firstMonth:** defaults to 0, this value is the first index of the months list to use. By default this means it will start with June from the first year in the range provided. Change to 1 to start with Aug instead, or to another number to start with another month in your custom month list.
- **lastMonth:** defaults to 1, this is the last index of the months list to use. By default this means it will end on Aug of the last year in the range provided. Change to 0 to end with June instead, or to another number to end with another month in your custom month list.
- **verbose:** defaults to False, set to True to print more information when running.

###### From unsorted images

If your images are not sorted, or you want to use all images in the directory for matching, you can use `DBUtil.get_unsorted_fish` instead.

```python
matchingImages = DBUtil.get_unsorted_fish(rootDirs=["path/to/images", "path/to/results"])
```
**rootDirs** is the same as in `DBUtil.get_fish`.

Other parameters:
- **excludeDirs:** can be used to provide a list of subdirectories to skip over.
- **verbose:** defaults to False, set to True to print more information when running.

###### From specific image paths

If you want only a select few images, you can provide a list of the paths to those images instead and use `DBUtil.get_fish_from_paths`

```python
queryImages = DBUtil.get_fish_from_paths(["path/to/IMG_1234.JPG", "path/to/IMG_2468.JPG"], rootDirs=["path/to/images", "path/to/results"])
```
Here, we use `DBUtil.get_fish_from_paths` to select two images for our query. We also need to provide the root image directories again for the image generation to work properly. This function also has the **verbose** option to provide more information when running.

##### Matching

Finally, we can run the matcher:
```python
matches = matcher.matching(query_images=queryImages, matching_imgs=matchingImages, rankingLimit=5)
```

Here I have passed in the two query images acquired from `DBUtil.get_fish_from_paths`. They are matched against the list of images acquired from either `DBUtil.get_fish` or `DBUtil.get_unsorted_fish`. We also set `rankingLimit` to 5 here to limit the results to the best 5 images.

This command can take quite a while to run. It will go through every image provided and generate the mask and spot files if they haven't already been created, then attempt to match then with the query image(s). Creating the mask and spot files takes some time and the matching takes some time itself. Subsequent runs will be considerably faster as the mask and spot files are saved, and the matching data is cached.

Other parameters:
- **algorithm:** this defaults to Algorithm.CUSTOM_GROTH. To use the Astroalign matcher pass in Algorithm.Astroalign instead.
- **verbose:** this defaults to False, set to True for more information when running. Using `verbose=True` is recommended when running the matcher so you can follow the progress at it runs and ensure it is running smoothly. Without this set it may seem like nothing is happening for a while.

The results from `matcher.matching` will be in the form of a list of results for each query. Each result is a dictionary with three elements:
- **"file_name":** The unique ID (UUID) of the fish file.
- **"ranking":** The ranking of the results. In this example with rankingLimit set to 5 this will a value from 1 to 5.
- **"score":** The 'score' given by the matching algorithm for the image. The higher the score, the more similar the image is to the query. Note that scores from the Groth matcher and Astroalign cannot be compared.

To get the `Fish` object from the UUID of a fish listed in the results use `DBUtil.get_fish_from_uuid`

```python
fishResult = DBUtil.get_fish_from_uuid("image-uuid-here", matchingImages)
```
The first parameter is the UUID of the fish, and the second is the list of Fish objects to search through. This function will return a single `Fish` object. The image path can then be attained from `fishResult.image_path`.

You can also display the image in Jupyter Notebook using IPython.display. This can be installed through pip by running `pip install ipython`.

```python
from IPython.display import Image
Image(fishResult.image_path)
```

This can also be used to display the mask and spot images by doing `Image(fishResult.mask_path)` or `Image(fishResult.spot_path)`