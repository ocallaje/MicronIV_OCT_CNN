from setuptools import setup, find_packages

setup(
    name="oct_segmentation",
    version="0.1.0",
    description="U-Net pipeline for mouse retinal OCT layer segmentation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "segmentation-models-pytorch>=0.3.3",
        "albumentations>=1.3.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0",
        "matplotlib>=3.7.0",
        "torchmetrics>=1.0.0",
    ],
)