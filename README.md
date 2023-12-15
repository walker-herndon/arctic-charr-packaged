## Re-identification of individuals from images using spot constellations; a case study in Arctic charr (Salvelinus alpinus)

Ignacy T. Dębicki, Elizabeth A. Mittell, Bjarni K. Kristjánsson, Camille A. Leblanc, Michael B. Morrissey, & Kasim Terzić

Modified by Walker Herndon to transform the code into a python library

#### Instructions for installation

You will need to have python and pip installed to use the library. You can check that these are installed by running `python —version` and `pip —version` in the terminal. Depending on how python was installed on your computer you might have to run `python3 —version` instead, and for any future python commands replace `python` with `python3`.

1. Run `python setup.py bdist_wheel` to build the distribution file (stored in the `dist` directory).

2. Create a virtual environment for working with the library and activate it. Further info on virtual environments is provided below if needed.

3. Run `pip install /path/to/wheelfile.whl` which installs the library and all dependencies.

4. Import as normal. To import the matcher for example, use `from arctic_charr_matcher.Matcher import Matcher` (imports `Matcher` class in `Matcher.py`). This could be done in a normal python file, or a Jupyter Notebook file. Note that to use Jupyter Notebook you will need to install it in the virtual environment with `pip install notebook`, then run with `jupyter notebook` and select your virtual environment as the kernel to use within Jupyter.

##### Virtual Environment Instructions
Using virtual environments when working with Python is good practice as it keeps your dependencies separated between different programs.

Comprehensive documentation of Python virtual environments, or “venv” can be found [here](https://docs.python.org/3/library/venv.html), but the basics are also provided below.

1. To create a new virtual environment run `python -m venv /path/to/venv` where the path can be anywhere you’d like to store the virtual environment. Virtual environment directories are often named something like `.venv` (the dot at the start makes the folder hidden by default), but you can name it whatever you’d like.
2. To use the venv, first activate it by running `source <venv>/bin/activate` where `<venv>` is the path to the venv given in step 1. This command assumes the shell you are using is bash or ash (bash is the default on Mac and Linux so if that doesn’t mean anything to you, you are almost definitely using bash). If you aren’t using bash or zsh the equivalent command for your shell can be found in the documentation linked above.
3. Now that the venv is activated, when you install a library using `pip install <library>` it will be installed within the venv instead of globally. This is why virtual environments are good to keep your dependencies organized.
4. To deactivate the virtual environment again, just run `deactivate`.

#### Usage
Start by importing the library
```python
from arctic_charr_matcher.Matcher import Matcher
from arctic_charr_matcher import DBUtil
```

Next, initialize the matcher
```python
matcher = Matcher(
	imgRoot="path/to/images",
	maskModelWeights="path/to/mask_model.hdf5",
	spotUnetWeights="path/to/spot_model.hdf5"
)
```

You'll need to supply the images to work with and the trained models.

Here's what the file structure for the images needs to look like:
<!-- Created with [tree.nathanfriend.io](https://tree.nathanfriend.io) -->

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
	├── 2012_August/
	├── 2013_June/
	├── 2013_August/
	└── ...
```

Next we need to get the image keys to pass into the matcher. For this we use `DBUtil`

```python
baseImages = DBUtil.get_images(21, rootDirs=["path/to/images", "path/to/results"], years=range(2015,2018))
```
Here we are loading in the base images. These are the images we will attempt to match the query against. There are a few parameters to consider:
- **Cave number:** I've picked 21 but it can be any cave you want to run the matcher on.
- **rootDirs:** should be the path to the images, and the path to a results directory which will be created when running the matcher if it hasn't already been. This directory will hold the image mask and spot files which is what the matcher actually runs on. It should be in the same root directory as you are running your script from.
- **years:** the range of years you want to include in `baseImages` (includes the start year but not the end, so my example is `[2015, 2016, 2017]`)

The same thing can be done to load in the query images, or if you just want to query a single image you can pass that in manually.

Finally, we can run the matcher:
```python
matches = matcher.matching(query_images=['C21-2019-June-IMG_3664'], matching_imgs=list(baseImages.keys()), rankingLimit=5)
```

Here I have just passed in a single query image: `images/2019_June/Cave21/IMG_3664.JPG`
`matching_imgs` takes in a list of keys which we get from the `baseImages` dict. We also set `rankingLimit` to 5 here to limit the results to the best 5 images.

This command can take quite a while to run. It will go through every image in `baseImages` and generate the mask and spot files if they haven't already been created, then attempt to match then with the query image. Creating the mask and spot files takes some time and the matching takes some time itself. Subsequent runs will be considerably faster as the mask and spot files are saved, and the matching data is cached.