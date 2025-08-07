"""
Setup script for Legal Contract Processing Pipeline
=================================================

Installation script for the contract processing pipeline package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="legal-contract-processor",
    version="1.0.0",
    author="AI Developer",
    author_email="developer@example.com",
    description="A comprehensive pipeline for analyzing legal contracts using Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/legal-contract-processor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Accounting",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.15.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.3",
            "torch>=1.12.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.15.0",
            "faiss-gpu>=1.7.3",
            "torch>=1.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "process-contracts=contract_processor:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="legal contracts nlp llm ai document-processing clause-extraction",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/legal-contract-processor/issues",
        "Source": "https://github.com/your-repo/legal-contract-processor",
        "Documentation": "https://github.com/your-repo/legal-contract-processor/wiki",
    },
)
