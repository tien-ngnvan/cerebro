from distutils.core import setup
from setuptools import find_packages

with open('README.md', encoding='utf-8') as file:
    description = file.read()

setup(
    name='cerebro',
    version='0.0.1',
    packages=find_packages(),
    license='Copyright (c) 2023 StableML Labs',
    zip_safe=True,
    description='Simple deep learning library which customized to our needs.',
    long_description=description,
    long_description_content_type='text/markdown',
    author='StableML Labs',
    author_email='hello@stableml.ai',
    url='https://gitlab.com/stableml/cerebro',
    keywords=[],
    install_requires=['accelerate', 'deepspeed'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)