# setup.py
from setuptools import setup, find_packages

setup(
    name="stephanie",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "stephanie": ["config/**/*.yaml",
        "prompts/**/*.txt",   # all prompt templates
        "schema.sql",  # your schema
        "docker-compose.yml",  # your schema
        ]
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
            "stephanie=main:main"
        ]
    }
)