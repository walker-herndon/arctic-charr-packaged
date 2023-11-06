from setuptools import setup, find_packages

setup(
    name="arctic_charr_matcher",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python-headless",
        "tensorflow",
        "keras",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "pandas",
        "xlrd",
        "sep",
    ],
    entry_points={
        "console_scripts": [
            "my_script=my_package.script:main",
        ],
    },
    author="Walker Herndon",
    author_email="walker.herndon01@gmail.com",
    description="Arctic Charr matching algorithm",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/walker-herndon/Arctic_charr_packaged",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
)
