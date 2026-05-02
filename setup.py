from setuptools import find_packages, setup


setup(
    name="hpc-stan",
    version="0.1.0",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.11",
    install_requires=[
        "bridgestan",
        "cmdstanpy",
        "dask[distributed]",
        "dask-jobqueue",
        "numpy",
        "pandas",
        "scipy",
    ],
    extras_require={
        "test": [
            "pytest",
        ],
    },
    author="Brian Parbhu",
    author_email="brian.parbhu@gmail.com",
    description="A Dask-powered HPC/cloud friendly interface for BridgeStan and CmdStanPy",
    license="GPL-3.0",
    keywords="stan hpc BridgeStan CmdStanPy SLURM Dask",
    url="https://github.com/bparbhu/hpc-stan",
)
