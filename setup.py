from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="aspect-pad",
    version="0.1.1", # Version bumped to bypass PyPI block
    author="Shin Thant Tun",
    description="A PyTorch-compatible transform for aspect-aware image scaling and dynamic mathematical padding.",
    long_description=long_description,
    long_description_content_type="text/markdown", # Tells PyPI to render it as Markdown
    packages=find_packages(),
    install_requires=[
        "Pillow",
    ],
    python_requires=">=3.7",
)