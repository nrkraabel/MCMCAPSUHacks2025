"""
Setup script for the Gerrymandering Simulator package.
"""

from setuptools import setup, find_packages

setup(
    name="gerrymandering_simulator",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "geopandas",
        "shapely",
        "pandas",
        "tqdm",
        "numba",
        "scipy",
    ],
    entry_points={
        'console_scripts': [
            'gerrymanderingsim=run_simulation:main',
        ],
    },
    author="Nicholas Kraabel",
    author_email="nrk5343@psu.edu",
    description="A simulator for visualizing the gerrymandering process",
    keywords="gerrymandering, simulation, redistricting, visualization",
    url="https://github.com/yourusername/gerrymandering_simulator",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
)