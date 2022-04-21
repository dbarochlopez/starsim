import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="starsim",                         # This is the name of the package
    version="3.0.1",                        # The initial release version
    author="David Baroch, Kike Herrero, Albert Rosich",                  # Full name of the author
    description="Spectroscopic and photometric simulation of a spotted rotating star.",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    install_requires=[
            'tqdm>=4.59.0',
            'numpy',
            'emcee>=3.1.0',
            'corner>=2.2.1',
            'configparser>=5.0.2',
            'matplotlib>3.2.0',
            'astropy>5.0',
            'scipy>=1.7.1',
            'numba>=0.53.1'
    ]                     # Install other dependencies if any
#dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0']

)

