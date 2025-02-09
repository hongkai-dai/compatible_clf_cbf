from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        subprocess.call(["pre-commit", "install"])


# Read the contents of requirements file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="compatible_clf_cbf",
    version="1.0",
    packages=find_packages(),
    install_requires=requirements,
    cmdclass={
        "install": PostInstallCommand,
    },
)
