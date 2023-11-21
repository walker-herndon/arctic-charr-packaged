## Re-identification of individuals from images using spot constellations; a case study in Arctic charr (Salvelinus alpinus)

Ignacy T. Dębicki, Elizabeth A. Mittell, Bjarni K. Kristjánsson, Camille A. Leblanc, Michael B. Morrissey, & Kasim Terzić

This branch is for transforming the code into a python library.

#### Instructions for installation
1. Run `python setup.py bdist_wheel` to build the distribution file (stored in the `dist` directory).
2. Create a virtual environment for working with the library and activate it.
3. Run `pip install /path/to/wheelfile.whl` which installs the library and all dependencies.
4. Import as normal. To import the matcher for example, use `from arctic_charr_matcher.Matcher import Matcher` (imports `Matcher` class in `Matcher.py`).

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
%% Created with [tree.nathanfriend.io](https://tree.nathanfriend.io)%%

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