from setuptools import setup

# Function to read the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]


# Read dependencies from requirements.txt
requirements = parse_requirements('requirements.txt')

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
