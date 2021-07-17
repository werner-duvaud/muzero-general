"""Setup script for muzero-baseline"""

import os.path
from setuptools import setup

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

# This call to setup() does all the work
setup(
    name="muzero-baseline",
    version="0.2.0",
    description="Baseline implementation of MuZero agent",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/LeoVS09/muzero-general",
    author="LeoVS09",
    author_email="leovs010@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=["muzero_baseline", "muzero_baseline.games"],
    include_package_data=True,
    install_requires=[       
        "numpy",
        "torch",
        "tensorboard",
        "gym",
        "ray",
        "seaborn",
        "nevergrad"
    ],
    entry_points={"console_scripts": ["muzero_baseline=muzero_baseline.muzero.__main__:main"]},
)