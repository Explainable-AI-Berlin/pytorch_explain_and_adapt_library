from setuptools import setup, find_packages
import setuptools

setuptools.setup(
    name="latent-diffusion",
    version="0.0.1",
    description="",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
    ],
)
