from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="data_preparation",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular pipeline for pattern discovery in time series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/data_preparation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'stumpy>=1.10.0',
        'matplotlib>=3.4.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'tqdm>=4.60.0',
        'pyyaml>=5.4.1'
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.0.0',
            'black>=21.0',
            'isort>=5.0.0',
            'mypy>=0.900',
        ],
    },
    entry_points={
        'console_scripts': [
            'data_prep=data_preparation.example_usage:main',
        ],
    },
)
