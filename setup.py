from os.path import abspath, dirname, join
from os import environ
from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))
with open(join(this_dir, "README.md"), encoding="utf-8") as file:
    long_description = file.read()


setup(
    name="dallecli",
    python_requires=">3.8",
    options={"bdist_wheel": {"universal": "1"}},
    version="2.1.0",
    description="A command line application to help wrap the OpenAI Dalle 3 api and other utilities.",
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
        "requests==2.28.2",
        "click==8.1.3",
        "openai==1.6.1",
        "rich==13.4.2",
        "idna",
        "pillow==9.4.0",
    ],
    entry_points={"console_scripts": ["dallecli=dallecli.cli:cli"]},
)
