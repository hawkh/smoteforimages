"""
Setup script for SMOTE Image Synthesis package.
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="smote-image-synthesis",
    version="0.1.0",
    description="SMOTE-based synthetic image generation system",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
    author="SMOTE Image Synthesis Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)