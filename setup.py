from setuptools import setup, find_packages

# Read the contents of requirements file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="compatible_clf_cbf",
    version="1.0",
    packages=find_packages(),
    install_requires=requirements,
)
