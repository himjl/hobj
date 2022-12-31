from setuptools import setup, find_packages
import os

if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as fb:
        requirements = fb.readlines()
else:
    requirements = []

print(find_packages())
setup(
    name="hobj",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    author="Michael J. Lee",
    author_email="mil@mit.edu",
    description="Human object learning benchmarks",
    keywords="",
)
