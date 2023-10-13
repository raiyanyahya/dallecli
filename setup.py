from os.path import abspath, dirname, join, isfile
from os import environ
from setuptools import find_packages, setup
import sys

this_dir = abspath(dirname(__file__))
with open(join(this_dir, "README.md"), encoding="utf-8") as file:
    long_description = file.read()


setup(
    name="dallecli",
    python_requires=">3.5",
    options={"bdist_wheel": {"universal": "1"}},
    version="1.3.0",
    description="A command line application to help wrap the OpenAI Dalle api and other utilities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raiyanyahya/dallecli",
    author="Raiyan Yahya",
    license="MIT",
    author_email="raiyanyahyadeveloper@gmail.com",
    keywords=[
        "cli",
        "developer tools",
        "productivity",
        "openai",
        "generative art",
        "ai",
    ],
    packages=find_packages(),
    install_requires=[
        "click==8.1.3",
        "openai==0.27.8",
        "rich==13.4.2",
        "idna",
        "pillow==9.4.0",
    ],
    entry_points={"console_scripts": ["dallecli=dallecli.cli:cli"]},
)
