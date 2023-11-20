## Re-identification of individuals from images using spot constellations; a case study in Arctic charr (Salvelinus alpinus)

Ignacy T. Dębicki, Elizabeth A. Mittell, Bjarni K. Kristjánsson, Camille A. Leblanc, Michael B. Morrissey, & Kasim Terzić

This branch is for transforming the code into a python library.

Instructions for installation:
1. Run `python setup.py bdist_wheel` to build the distribution file (stored in the `dist` directory).
2. Create a virtual environment for working with the library and activate it.
3. Run `pip install /path/to/wheelfile.whl` which installs the library and all dependencies.
4. Import as normal. To import the matcher for example, use `from arctic_charr_matcher.Matcher import Matcher` (imports `Matcher` class in `Matcher.py`).