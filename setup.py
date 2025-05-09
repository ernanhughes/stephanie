# setup.py
from setuptools import setup, find_packages

setup(
    name="co_ai",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "co_ai": ["config/**/*.yaml"],
    },
    install_requires=[
        "omegaconf",
        "hydra-core",
        "psycopg2-binary",
        "numpy",
        "dspy",
    ],
    entry_points={
        "console_scripts": [
            "co_ai=main:main"
        ]
    }
)