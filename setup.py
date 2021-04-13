import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tiresias",
    version="0.0.1",
    author="Anonymized",
    author_email="anonymized@anonymized.edu",
    description="Pytorch implementation of Tiresias: Predicting Security Events Through Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Anonymized/Tiresias",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
