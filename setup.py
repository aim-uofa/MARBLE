from setuptools import setup, find_packages

setup(
    name="diffusion-nft",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
    ],
    extras_require={
        "dev": [
            "ipython==8.34.0",
            "black==24.2.0"
        ]
    }
)
