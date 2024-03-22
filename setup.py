from setuptools import setup, find_packages
import os 
from pathlib import Path 

REPO_NAME = 'my_project'
AUTHOR_USER_NAME = 'HarishkumarSedu'
SRC_REPO = 'mlapp'
AUTHOR_EMAIL = 'harishkumarSedu@gmail.com'

version_filepath = 'config/Version.txt'
long_description_filepath = f'src/{SRC_REPO}/README.md'
# check the for wversion file 
if os.path.exists(version_filepath := Path(version_filepath)) :
    with open(version_filepath, 'r')  as version_file :
        __VERSION__ = version_file.read()
else:
    __VERSION__ = '0.0.0'


if os.path.exists(long_description_filepath := Path(long_description_filepath)) :
    with open(long_description_filepath, 'r')  as description_file :
        long_description = description_file.read()

setup(
    name=REPO_NAME,
    author=AUTHOR_USER_NAME,
    version=__VERSION__,
    description="A small python package for ml app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                ],

)