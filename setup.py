from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="setfit_ig",
    description="SetFit + Integrated gradients.",
    license="MIT",
    packages=find_packages(include="setfit_ig"),
    install_requires=requirements,
)
