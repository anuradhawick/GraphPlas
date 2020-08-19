import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GraphPlas", # Replace with your own username
    version="0.1-rc",
    author="Anuradha Wickramarachchi",
    author_email="anuradhawick@gmail.com",
    description="GraphPlas: Assembly Graph Assisted Recovery of Plasmidic Contigs from NGS Assemblies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anuradhawick/GraphPlas",
    packages=setuptools.find_packages(),
    scripts=['GraphPlas.py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "python-igraph",
        "biopython",
        "tqdm",
        "tabulate",
        "setuptools"],
    python_requires='>=3.6',
)