"""
Setup script for Ising Field Theory Quantum Simulation package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ising-field-theory",
    version="1.0.0",
    author="Quantum Simulation Team",
    author_email="",
    description="Quantum simulation of Ising Field Theory for scattering and particle production studies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ising-field-theory",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
        "viz": [
            "matplotlib>=3.4.0",
        ],
    },
    keywords=[
        "quantum computing",
        "ising model",
        "field theory",
        "scattering",
        "quantum simulation",
        "E8",
        "integrable systems",
    ],
)
