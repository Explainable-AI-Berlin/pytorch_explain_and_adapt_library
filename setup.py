import setuptools

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setuptools.setup(
    name='peal',
    version='0.1',
    packages=['peal'],
    install_requires=required_packages,
    setup_requires=['setuptools>=38.6.0'],
)
