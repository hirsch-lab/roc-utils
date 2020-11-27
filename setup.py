"""Package build and install script.

Useful commands:
    python setup.py clean                 Clean temporary files
    python setup.py sdist                 Create source distribution (.tar.gz)
    python setup.py bdist_wheel           Create built distribution (.whl)
    python setup.py sdist bdist_wheel     Create both
    python setup.py flake8                Run flake8 (coding style check)
    pip install dist/roc_utils...tar.gz   Install from local tarball
    pip show roc_utils                    Show package information
    pip uninstall roc_utils               Uninstall
    twine check dist/*                    Check the markup in the README
    twine upload --repository testpypi dist/* Upload everything to TestPyPI
    pip install --index-url https://test.pypi.org/simple/ --no-deps roc_utils
"""

import sys
import numpy
from pathlib import Path
from setuptools import setup, find_packages


def get_version():
    """Loads version from VERSION file. Raises an IOError on failure."""
    with open("VERSION", "r") as fid:
        ret = fid.read().strip()
    return ret


def get_readme():
    """Load README.rst for display on PyPI."""
    with open("README.rst") as fid:
        return fid.read()


install_packages = ["numpy", "scipy", "pandas", "matplotlib"]
testing_packages = ["scikit-learn"] + install_packages

cmdclass = {}
subcommand = sys.argv[1] if len(sys.argv) > 1 else None
if subcommand == "build_ext":
    # This requires Cython. We come here if the extension package is built.
    from Cython.Distutils import build_ext
    # To get some HTML output with an overview of the generate C code.
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True
    # Cython src.
    miniball_src = ["bindings/_miniball_wrap.pyx"]
    cmdclass["build_ext"] = build_ext
else:
    # This uses the "pre-compiled" Cython output.
    miniball_src = ["bindings/_miniball_wrap.cpp"]

include_dirs = [Path(__file__).parent.absolute(),
                numpy.get_include()]

with open("README.md", encoding="utf-8") as fid:
    long_description = fid.read()

setup(name="roc_utils",
      version=get_version(),
      url="https://github.com/hirsch-lab/roc-utils",
      author="Norman Juchler",
      description=("Tools to compute and visualize ROC curves."),
      long_description=long_description,
      long_description_content_type="text/markdown",
      license="MIT",
      keywords="ROC AUC receiver operating characteristic",
      packages=find_packages(),
      python_requires=">=3.6",
      install_requires=install_packages,
      tests_require=testing_packages,
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Software Development :: Libraries",
            "Topic :: Utilities",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
      ],
      # setup_requires=["flake8"]
      )
