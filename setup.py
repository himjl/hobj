from setuptools import setup

# Read dependencies from requirements.txt
with open('requirements.txt', 'r') as file:
    requirements = [line.strip() for line in file if line.strip() and not line.startswith('#')]

setup(
    name="hobj",
    version="2.0.0",
    packages=['hobj'],
    install_requires=requirements,
    author="Michael J. Lee",
    author_email="mil@mit.edu",
    description="Human object learning benchmarks",
    keywords="",
    python_requires='>=3.11',
)
