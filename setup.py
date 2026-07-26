from setuptools import setup, find_packages

setup(
    name="aspect-pad",
    version="0.1.0",
    author="Shin Thant Tun",
    description="A PyTorch-compatible transform for aspect-aware image scaling and dynamic mathematical padding.",
    packages=find_packages(),
    install_requires=[
        "Pillow",
        # "torchvision" # (Optional: uncomment if you want to strictly require PyTorch)
    ],
    python_requires=">=3.7",
)