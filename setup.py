from setuptools import setup, find_packages

setup(
    name="neurode",
    version="0.1.0",
    author="wokron",
    author_email="stringcatwok@gmail.com",
    description="neurode is a framework for solving and fitting ODEs",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch -i https://download.pytorch.org/whl/cpu",
        "scipy",
        "tqdm",
        "numpy",
    ],
)
