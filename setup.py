from setuptools import setup

setup(
    name="pomap",
    version="0.0",
    description="A Functor for building complex ML models on Polars DataFrames",
    author="Niklas Mather",
    author_email="niksmather@gmail.com",
    packages=["pomap"],
    install_requires=["polars"],  # external packages as dependencies
)
