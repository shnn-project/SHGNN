#!/usr/bin/env python3
"""
Setup script for SHNN Python package.

This setup.py provides both source distribution and wheel building
capabilities for the SHNN (Spiking Hypergraph Neural Network) library.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

# Read version from Cargo.toml
def get_version():
    """Extract version from Cargo.toml in the Python crate."""
    cargo_toml = Path("crates/shnn-python/Cargo.toml")
    if cargo_toml.exists():
        with open(cargo_toml, 'r') as f:
            for line in f:
                if line.startswith('version = '):
                    return line.split('"')[1]
    return "0.1.0"

# Read long description from README
def get_long_description():
    """Read the long description from README.md."""
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "SHNN - Spiking Hypergraph Neural Network Library"

# Read requirements from requirements.txt
def get_requirements():
    """Read requirements from requirements.txt."""
    req_path = Path("requirements.txt")
    requirements = []
    
    if req_path.exists():
        with open(req_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Skip optional dependencies that might not be available
                    if not any(skip in line for skip in ['cupy', 'pyopencl', 'pytest', 'sphinx', 'black', 'isort', 'mypy']):
                        requirements.append(line)
    
    # Core requirements that must be present
    core_requirements = [
        "numpy>=1.21.0",
        "typing-extensions>=4.0.0",
    ]
    
    # Add core requirements if not already present
    for req in core_requirements:
        pkg_name = req.split('>=')[0].split('==')[0]
        if not any(pkg_name in existing for existing in requirements):
            requirements.append(req)
    
    return requirements

# Check if we're building from source or installing
def is_building_from_source():
    """Check if we're in the source directory."""
    return Path("crates/shnn-python").exists()

# Rust extension configuration
def get_rust_extensions():
    """Configure Rust extensions if building from source."""
    if not is_building_from_source():
        return []
    
    return [
        RustExtension(
            "shnn._shnn_python",
            path="crates/shnn-python/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
            features=["extension-module"],
        )
    ]

# Package configuration
setup(
    name="shnn-python",
    version=get_version(),
    author="SHNN Team",
    author_email="team@shnn-project.org",
    description="Python bindings for Spiking Hypergraph Neural Networks",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/shnn-project/shnn",
    project_urls={
        "Homepage": "https://shnn-project.github.io",
        "Documentation": "https://shnn-python.readthedocs.io",
        "Repository": "https://github.com/shnn-project/shnn",
        "Bug Tracker": "https://github.com/shnn-project/shnn/issues",
        "Changelog": "https://github.com/shnn-project/shnn/blob/main/CHANGELOG.md",
    },
    
    # Package discovery
    packages=find_packages(where="python", exclude=["tests*"]) if Path("python").exists() else ["shnn"],
    package_dir={"": "python"} if Path("python").exists() else {},
    
    # Rust extensions
    rust_extensions=get_rust_extensions(),
    
    # Dependencies
    install_requires=get_requirements(),
    
    # Optional dependencies
    extras_require={
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0", 
            "plotly>=5.0.0",
        ],
        "ml": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "scikit-learn>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "tqdm>=4.62.0",
        ],
        "cuda": [
            "cupy>=10.0.0",
        ],
        "opencl": [
            "pyopencl>=2021.2.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            "matplotlib>=3.5.0", "seaborn>=0.11.0", "plotly>=5.0.0",
            "torch>=1.9.0", "torchvision>=0.10.0", "scikit-learn>=1.0.0",
            "jupyter>=1.0.0", "ipywidgets>=7.6.0", "tqdm>=4.62.0",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for PyPI
    keywords=[
        "neural-networks", "neuromorphic", "spiking", "machine-learning",
        "rust", "python", "performance", "hardware-acceleration"
    ],
    
    # Entry points
    entry_points={
        "console_scripts": [
            "shnn=shnn.cli:main",
        ],
    },
    
    # Include package data
    include_package_data=True,
    zip_safe=False,
    
    # Build configuration
    setup_requires=["setuptools-rust>=1.5.0"] if is_building_from_source() else [],
)