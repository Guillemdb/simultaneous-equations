from importlib.machinery import SourceFileLoader
from pathlib import Path

from setuptools import find_packages, setup


version = SourceFileLoader(
    "semtorch.version",
    str(Path(__file__).parent / "src" / "semtorch" / "version.py"),
).load_module()

with open(Path(__file__).with_name("README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Module-specific dependencies.
extras = {
    "dataviz": [
        "matplotlib",
        "bokeh>2.0.0",
        "pandas",
        "panel",
        "holoviews",
        "hvplot",
        "plotly",
        "streamz",
        "param",
        "selenium",
        "pyarrow",
    ],
    "test": ["pytest>=5.3.5", "hypothesis>=5.6.0"],
    "ray": ["ray>=1.0.1.post1", "setproctitle"],
}

# Meta dependency groups.
extras["all"] = [item for group in extras.values() for item in group]

setup(
    name="semtorch",
    description="Simultaneous equation models in pytorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    version=version.__version__,
    license="MIT",
    author="Guillem Duran Ballester",
    author_email="info@fragile.tech",
    url="https://github.com/FragileTech/fragile",
    download_url="https://github.com/FragileTech/fragile",
    keywords=["Simultaneous equation models", "sem"],
    tests_require=["pytest>=5.3.5", "hypothesis>=5.6.0"],
    extras_require=extras,
    install_requires=[
        "flogging",
        "numba",
        "scipy",
        "tqdm",
    ],
    package_data={"": ["README.md"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
    ],
)
