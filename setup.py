from setuptools import setup, find_packages
import os

# 读取 requirements.txt
install_requires = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("@"):
                install_requires.append(line)

setup(
    name="deelogeny_m2",
    version="0.1",
    packages=find_packages(where='src'),
    package_dir={"":"src"},
    install_requires=install_requires,
    python_requires=">=3.10",
    author="Master Bioinfo@Lyon",
    description="Learning-based analysis of the realism of phylogenetic simulation methods",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # autres options
)
